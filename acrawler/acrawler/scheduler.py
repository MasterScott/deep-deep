# -*- coding: utf-8 -*-
"""
There are several conflicting goals which scheduler needs to acheive.
It should:

1. prefer to crawl all domains;
2. prefer more promising links;
3. allow less promising links to be crawled with some probability
   (ε-greedy policy);
4. allow to update link priorities dynamically.

"""
import heapq
import itertools

import numpy as np
# from twisted.internet.task import LoopingCall

from acrawler.score_pages import available_form_types
from scrapy.utils.misc import load_object
from acrawler.utils import (
    dict_subtract,
    dict_aggregate_max,
    softmax,
    get_response_domain
)


class RequestsPriorityQueue:
    """
    In-memory priority queue for requests.

    Unlike default Scrapy queues it supports high-cardinality priorities
    (but no float priorities becase scrapy.Request doesn't support them).

    This queue allows to change request priorities. To do it

    1. iterate over queue.entries;
    2. call queue.change_priority(entry, new_priority) for each entry;
    3. call queue.heapify()

    """
    def __init__(self, fifo=True):
        self.entries = []
        step = 1 if fifo else -1
        self.counter = itertools.count(step=step)

    def push(self, request):
        count = next(self.counter)
        entry = [-request.priority, count, request]
        heapq.heappush(self.entries, entry)
        return entry

    def pop(self):
        if not self.entries:
            return None
            # raise KeyError("queue is empty")
        priority, count, request = heapq.heappop(self.entries)
        return request

    @classmethod
    def change_priority(cls, entry, new_priority):
        """
        Change priority of an existing entry.

        ``entry`` is an item from :attr:`entries` attribute.

        After priorities are changed it is necessary to call
        :meth:`heapify`.
        """
        entry[0] = -new_priority
        entry[2].priority = new_priority

    @classmethod
    def get_priority(cls, entry):
        return -entry[0]

    def heapify(self):
        heapq.heapify(self.entries)

    # def iter_requests(self):
    #     """
    #     Return all Request objects in a queue.
    #     The first request is guaranteed to have top priority;
    #     order of other requests is arbitrary.
    #     """
    #     return (e[2] for e in self.entries)

    def __len__(self):
        return len(self.entries)


FLOAT_PRIORITY_MULTIPLIER = 10000


def get_request_predicted_scores(request, G):
    node_id = request.meta.get('node_id')
    if node_id is None:
        return

    node = G.node[node_id]
    assert not node['visited']
    return node['predicted_scores']


class DomainFormFinderRequestsQueue(RequestsPriorityQueue):
    def __init__(self, domain, form_types, G):
        super().__init__(fifo=True)
        self.domain = domain
        self.max_observed_scores = {tp: 0 for tp in form_types}
        self.G = G

    @property
    def weight(self):
        if not self.entries:
            return -1000 * FLOAT_PRIORITY_MULTIPLIER
        top_priority = self.get_priority(self.entries[0])
        return top_priority

    def push(self, request):
        request.priority = self.compute_priority(request)
        return super().push(request)

    def compute_priority(self, request):
        """ Return request priority based on its scores """
        scores = get_request_predicted_scores(request, self.G)
        if scores is None:
            if self.domain is None:
                reward = 100  # seed URLs
            else:
                reward = 0.1  # no classifier yet
        else:
            rewards = dict_subtract(scores, self.max_observed_scores)
            reward = max(max(rewards.values()), 0)
        return int(reward * FLOAT_PRIORITY_MULTIPLIER)

    def update_observed_scores(self, observed_page_scores):
        new_max = dict_aggregate_max(self.max_observed_scores,
                                     observed_page_scores)
        if new_max != self.max_observed_scores:
            scores_diff = {
                k: v
                for k, v in dict_subtract(new_max, self.max_observed_scores).items()
                if v
            }
            print("Max scores updated for {}. Diff: {}".format(
                self.domain, scores_diff
            ))
            self.max_observed_scores = new_max
            self.recalculate_priorities()

    def recalculate_priorities(self):
        """
        Update all request priorities.
        It can be necessary in 2 cases:

        1. predicted request scores are changed, or
        2. max_scores are changed.
        """
        for entry in self.entries:
            request = entry[2]
            self.change_priority(entry, self.compute_priority(request))
        self.heapify()

    def __repr__(self):
        return "DomainRequestQueue({}; #requests={}, weight={})".format(
            self.domain, len(self), self.weight
        )


class BalancedPriorityQueue:
    """ This queue samples other queues randomly, based on their weights """
    def __init__(self, form_types, G):
        self.G = G
        self.form_types = form_types
        self.queues = {}  # domain -> queue

        # self.gc_task = LoopingCall(self._gc)
        # self.gc_task.start(60, now=False)

    # def _gc(self):
    #     pass

    def push(self, request):
        domain = request.meta.get('domain')
        if domain not in self.queues:
            self.queues[domain] = DomainFormFinderRequestsQueue(
                domain, self.form_types, self.G
            )
        self.queues[domain].push(request)

    def pop(self):
        domains = list(self.queues.keys())
        if not domains:
            return
        weights = [self.queues[domain].weight for domain in domains]
        p = softmax(weights, t=FLOAT_PRIORITY_MULTIPLIER)
        queue = self.queues[np.random.choice(domains, p=p)]
        # print(queue, dict(zip(domains, p)))
        req = queue.pop()
        if req:
            scores = get_request_predicted_scores(req, self.G)
            if scores:
                scores = {k: int(v*100) for k, v in scores.items()}
            print(req.priority, scores, req.url)

        return req

    def update_observed_scores(self, response, observed_scores):
        domain = get_response_domain(response)
        if domain not in self.queues:
            return
        self.queues[domain].update_observed_scores(observed_scores)


class Scheduler:
    def __init__(self, dupefilter, stats):
        self.dupefilter = dupefilter
        self.stats = stats
        self.queue = None
        self.spider = None

    @classmethod
    def from_crawler(cls, crawler):
        settings = crawler.settings
        dupefilter_cls = load_object(settings['DUPEFILTER_CLASS'])
        dupefilter = dupefilter_cls.from_settings(settings)
        return cls(
            dupefilter=dupefilter,
            stats=crawler.stats,
        )

    def has_pending_requests(self):
        return len(self.queue) > 0

    def open(self, spider):
        self.spider = spider
        self.queue = BalancedPriorityQueue(
            form_types=available_form_types(),
            G=spider.G,
        )
        return self.dupefilter.open()

    def close(self, reason):
        return self.dupefilter.close(reason)

    def enqueue_request(self, request):
        if not request.dont_filter:
            if self.dupefilter.request_seen(request):
                self.dupefilter.log(request, self.spider)
                return False

        self.queue.push(request)
        self.stats.inc_value('custom-scheduler/enqueued/', spider=self.spider)
        return True

    def next_request(self):
        request = self.queue.pop()
        if request:
            self.stats.inc_value('custom-scheduler/dequeued/', spider=self.spider)
        return request
