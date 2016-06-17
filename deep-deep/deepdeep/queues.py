# -*- coding: utf-8 -*-
"""
Queues
======

There are several conflicting goals which scheduler needs to acheive.
It should:

1. prefer to crawl all domains;
2. prefer more promising links;
3. allow less promising links to be crawled with some probability
   (ε-greedy policy);
4. allow to update link priorities dynamically.

This module contains custom Scrapy queues which allow to do that:
:class:`BalancedPriorityQueue` allows to have per-domain request queues and
sample from them, :class:`RequestsPriorityQueue` is a per-domain queue
which allows to update request priorities.
"""
from typing import Dict
from typing import Set

import heapq
import itertools
import random
from typing import List, Any, Iterable, Optional, Callable

import numpy as np
# from twisted.internet.task import LoopingCall
import scrapy
from deepdeep.utils import softmax


FLOAT_PRIORITY_MULTIPLIER = 10000


def score_to_priority(score: float) -> int:
    return int(score * FLOAT_PRIORITY_MULTIPLIER)


def priority_to_score(prio: int) -> float:
    return prio / FLOAT_PRIORITY_MULTIPLIER


class QueueClosed(Exception):
    pass


class RequestsPriorityQueue:
    """
    In-memory priority queue for requests.

    Unlike default Scrapy queues it supports high-cardinality priorities
    (but no float priorities becase scrapy.Request doesn't support them).

    This queue allows to change request priorities. To do it

    1. iterate over queue.entries;
    2. call queue.change_priority(entry, new_priority) for each entry;
    3. call queue.heapify()

    It also allows to remove a request from a queue using remove_entry.
    """

    REMOVED = object()

    REMOVED_PRIORITY = score_to_priority(10000)
    EMPTY_PRIORITY = score_to_priority(-10000)

    def __init__(self, fifo: bool=True) -> None:
        # entries are lists of [int, int, scrapy.Request]
        self.entries = []  # type: List[List]
        step = 1 if fifo else -1
        self.counter = itertools.count(step=step)

    def push(self, request: scrapy.Request) -> List:
        count = next(self.counter)
        entry = [-request.priority, count, request]
        heapq.heappush(self.entries, entry)
        return entry

    def pop(self) -> Optional[scrapy.Request]:
        while self.entries:
            priority, count, request = heapq.heappop(self.entries)
            if request is not self.REMOVED:
                return request
        # raise KeyError('pop from an empty priority queue')

    @classmethod
    def change_priority(cls,
                        entry: List,
                        new_priority: int) -> None:
        """
        Change priority of an existing entry.

        ``entry`` is an item from :attr:`entries` attribute.

        After priorities are changed it is necessary to call
        :meth:`heapify`.
        """
        entry[0] = -new_priority
        if cls.entry_is_active(entry):
            entry[2].priority = new_priority

    @classmethod
    def entry_is_active(cls, entry: List) -> bool:
        return entry[2] is not cls.REMOVED

    def iter_active_entries(self) -> Iterator[List]:
        return (e for e in self.entries if self.entry_is_active(e))

    def update_all_priorities(self,
                              compute_priority_func: Callable[[List[scrapy.Request]], List[int]]) -> None:
        """
        Update all request priorities.

        ``compute_priority_func`` is a function which returns
        new priority; it should accept a list of Requests and return a list of
        integer priorities.
        """
        requests = list(self.iter_requests())
        new_priorities = compute_priority_func(requests)
        for entry, priority in zip(self.iter_active_entries(), new_priorities):
            self.change_priority(entry, priority)
        self.heapify()

    def remove_entry(self, entry: List) -> scrapy.Request:
        """
        Mark an existing entry as removed.
        ``entry`` is an item from :attr:`entries` attribute.
        """
        request = entry[2]
        entry[2] = self.REMOVED
        # move removed entry to the top at next heapify call
        max_prio = 0 if not self.entries else -self.entries[0][0]
        entry[0] = - (max_prio + self.REMOVED_PRIORITY)
        return request

    def pop_random(self, n_attempts: int=10) -> Optional[scrapy.Request]:
        """ Pop random entry from a queue """
        self._pop_empty()
        if not self.entries:
            return

        # Because we've called _pop_empty it is guaranteed there is at least
        # one non-removed entry in a queue (the one at the top).
        for i in range(n_attempts):
            entry = random.choice(self.entries)
            if entry[2] is not self.REMOVED:
                request = self.remove_entry(entry)
                return request

    def max_priority(self) -> int:
        """ Return maximum request priority in this queue """
        if not self.entries:
            return self.EMPTY_PRIORITY
        top_priority = self.get_priority(self.entries[0])
        return top_priority

    @property
    def next_request(self) -> Optional[scrapy.Request]:
        if not self.entries:
            return None
        return self.entries[0][2]

    @classmethod
    def get_priority(cls, entry) -> int:
        return -entry[0]

    def heapify(self) -> None:
        heapq.heapify(self.entries)
        self._pop_empty()

    def _pop_empty(self):
        """ Pop all removed entries from heap top """
        while self.entries and self.next_request is self.REMOVED:
            heapq.heappop(self.entries)

    def iter_requests(self) -> Iterable[scrapy.Request]:
        """
        Return all Request objects in a queue.
        The first request is guaranteed to have top priority;
        order of other requests is arbitrary.
        """
        return (e[2] for e in self.iter_active_entries())

    def __len__(self) -> int:
        return len(self.entries)


class BalancedPriorityQueue:
    """
    This queue samples other queues randomly, based on their weights
    (i.e. based on top request priority in a given queue).

    "Bins" to balance should be set in ``request.meta['scheduler_slot']``.
    For each ``scheduler_slot`` value a separate queue is created.

    queue_factory should be a function which returns a new
    RequestsPriorityQueue for a given slot name.

    ``eps`` is a probability of choosing random queue and
    returning random request from it. Because sampling is two-stage,
    it is biased towards queues with fewer requests.

    ``balancing_temperature`` is a parameter which controls how to
    choose the queue to get requests from. If the value is high,
    queue will be selected almost randomly. If the value is close to zero,
    queue with a highest request priority will be selected with a probability
    close to 1. Default value is 1.0; it means queues are selected randomly
    with probabilities proportional to max priority of their requests.
    """
    def __init__(self,
                 queue_factory: Callable[[str], RequestsPriorityQueue],
                 eps: float=0.0,
                 balancing_temperature: float=1.0) -> None:
        assert balancing_temperature > 0
        self.queues = {}  # type: Dict[str, Optional[RequestsPriorityQueue]]
        self.closed_slots = set()  # type: Set[str]
        self.eps = eps
        self.queue_factory = queue_factory
        self.balancing_temperature = balancing_temperature

    def push(self, request: scrapy.Request) -> None:
        slot = request.meta.get('scheduler_slot')
        if slot in self.closed_slots:
            raise QueueClosed()
        if slot not in self.queues:
            self.queues[slot] = self.queue_factory(slot)
        self.queues[slot].push(request)

    def pop(self) -> Optional[scrapy.Request]:
        keys = list(self.queues.keys())
        if not keys:
            return

        random_policy = self.eps and random.random() < self.eps
        # if random_policy:
        #     print("ε", end=' ')

        if random_policy:
            queue = self.queues[random.choice(keys)]
        else:
            weights = [self.queues[key].max_priority() for key in keys]
            temperature = FLOAT_PRIORITY_MULTIPLIER * self.balancing_temperature
            p = softmax(weights, t=temperature)
            queue = self.queues[np.random.choice(keys, p=p)]
        # print(queue, dict(zip(domains, p)))
        request = queue.pop_random() if random_policy else queue.pop()
        if request is not None:
            request.meta['from_random_policy'] = random_policy
        return request

    def get_active_slots(self) -> List[str]:
        return [key for key, queue in self.queues.items() if queue]

    def get_queue(self, slot: str):
        return self.queues[slot]

    def close_queue(self, slot: str) -> int:
        """
        Close a queue. Requests for this queue are dropped,
        including requests which are already scheduled.

        Return a number of dropped requests.
        """
        self.closed_slots.add(slot)
        queue = self.queues.pop(slot, None) or []
        return len(queue)

    def iter_active_requests(self) -> Iterable[scrapy.Request]:
        """ Return an iterator over all requests in a queue """
        for q in self.queues.values():
            yield from q.iter_requests()

    def __len__(self) -> int:
        return sum(len(q) for q in self.queues.values() if q)
