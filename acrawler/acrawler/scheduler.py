# -*- coding: utf-8 -*-
"""
Scheduling
==========

There are several conflicting goals which scheduler needs to acheive.
It should:

1. prefer to crawl all domains;
2. prefer more promising links;
3. allow less promising links to be crawled with some probability
   (ε-greedy policy);
4. allow to update link priorities dynamically.

For each domain there is a priority queue for requests.
To select next link crawler first chooses a domain, then chooses
a link for this domain. It allows to crawl all domains and prioritise
more promising requests.

Choosing links to follow for a domain
-------------------------------------

Request priorities are calculated based on link rewards.

Reward is a difference between current max score for a task (for a form type)
and expected max score for a task (for a form type). For example, if max
probability of a search form on any of the pages from this domain so far
was 0.2, and there is a link which leads to a search form with
probability 0.8, the reward for this link is 0.8-0.2=0.6.

Request priority is computed as a maximum of all expected rewards for this
request.

The approach above has a few pathological cases:

1. Links scores are probabilities of finding a form of a given class
   on a target page. But we don't know for sure if there is indeed a form
   of a given class on a target page even when we observed the target page:
   form classifier is probabilistic. Currently link scores are defined as
   ``P( target_score > 0.5 | link)``. It means that they are biased.
   For example, if target score is always 0.6, link score will be pushed
   to 1.0, and an expected reward will be always positive (0.4).
   Optimal model would assign zero reward in this case, i.e. link scores should
   predict target scores, not solve a binary classification problem.

   To tackle this, rewards are computed for adjusted scores: if score > 0.5
   it is set to 1.0, otherwise it is set to 0.0. A better approach would be
   to use cross-entropy as training objective for link classifier (TODO/FIXME).

2. If some kind of forms is common and many links lead to these forms
   then link scores would be high for this kind of forms for most links.
   This is a problem if some domain doesn't have these forms at all.
   Example: let's say 80% of websites have search forms on each page.
   So classifier learned to assign each link 0.8 score on average.
   If a domain doesn't have a search form then the reward for most links
   will be a high number of 0.8, and so crawler can spend most time trying
   to find a search form instead of trying to solve other tasks.

Choosing domain to crawl
------------------------

For each domain crawler maintains a score - max expected reward.
Next domain to crawl is selected randomly, with a probability proportional
to this score - more promising domain is, more often it is selected.

"""
import heapq
import itertools
import random

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


FLOAT_PRIORITY_MULTIPLIER = 10000


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

    REMOVED = object()

    REMOVED_PRIORITY = 1000 * FLOAT_PRIORITY_MULTIPLIER
    EMPTY_PRIORITY = -1000 * FLOAT_PRIORITY_MULTIPLIER

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
        while self.entries:
            priority, count, request = heapq.heappop(self.entries)
            if request is not self.REMOVED:
                return request
        # raise KeyError('pop from an empty priority queue')

    @classmethod
    def change_priority(cls, entry, new_priority):
        """
        Change priority of an existing entry.

        ``entry`` is an item from :attr:`entries` attribute.

        After priorities are changed it is necessary to call
        :meth:`heapify`.
        """
        entry[0] = -new_priority
        if entry[2] is not cls.REMOVED:
            entry[2].priority = new_priority

    def update_all_priorities(self, compute_priority_func):
        """
        Update all request priorities.

        ``compute_priority_func`` is a function which returns
        new priority; it should accept a Request and return an integer.
        """
        for entry in self.entries:
            request = entry[2]
            if request is not self.REMOVED:
                priority = compute_priority_func(request)
                self.change_priority(entry, priority)
        self.heapify()

    def remove_entry(self, entry):
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

    def pop_random(self, n_attempts=10):
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

    def max_priority(self):
        """ Return maximum request priority in this queue """
        if not self.entries:
            return self.EMPTY_PRIORITY
        top_priority = self.get_priority(self.entries[0])
        return top_priority

    @classmethod
    def get_priority(cls, entry):
        return -entry[0]

    def heapify(self):
        heapq.heapify(self.entries)
        self._pop_empty()

    def _pop_empty(self):
        """ Pop all removed entries from heap top """
        while self.entries and self.entries[0][2] is self.REMOVED:
            heapq.heappop(self.entries)

    def iter_requests(self):
        """
        Return all Request objects in a queue.
        The first request is guaranteed to have top priority;
        order of other requests is arbitrary.
        """
        return (e[2] for e in self.entries if e[2] != self.REMOVED)

    def __len__(self):
        return len(self.entries)


def get_request_predicted_scores(request, G):
    """ Return stored predicted scores for a request """
    node_id = request.meta.get('node_id')
    if node_id is None:
        return

    node = G.node[node_id]
    assert not node['visited']
    return node['predicted_scores']


class DomainFormFinderRequestsQueue(RequestsPriorityQueue):
    def __init__(self, domain, form_types, G, zeroone_loss):
        super().__init__(fifo=True)
        self.domain = domain
        self.max_observed_scores = {tp: 0 for tp in form_types}
        self.G = G
        self.zeroone_loss = zeroone_loss

    def push(self, request):
        request.priority = self.compute_priority(request)
        return super().push(request)

    def compute_priority(self, request):
        """ Return request priority based on its scores """
        scores = get_request_predicted_scores(request, self.G)
        if scores is None:
            if self.domain is None:
                expected_reward = 100  # seed URLs
            else:
                expected_reward = 0.1  # no classifier yet
        else:
            expected_rewards = dict_subtract(scores, self.max_observed_scores)
            expected_reward = max(max(expected_rewards.values()), 0)
        return int(expected_reward * FLOAT_PRIORITY_MULTIPLIER)

    def update_observed_scores(self, observed_page_scores):
        if self.zeroone_loss:
            # use the same loss for reward and for link classifier
            # FIXME: it should be handled from the other side, we should
            # train classifier with cross-entropy objective
            observed_page_scores = {tp: 1.0 if v > 0.5 else 0.0
                                    for tp, v in observed_page_scores.items()}
        new_max = dict_aggregate_max(
            self.max_observed_scores,
            observed_page_scores
        )
        if new_max != self.max_observed_scores:
            scores_diff = sorted([
                (v, tp)
                for tp, v in dict_subtract(new_max, self.max_observed_scores).items()
                if v > 0.01
            ], reverse=True)

            scores_diff_repr = ", ".join([
                "{} +{:0.2f}".format(tp, v)
                for v, tp in scores_diff
            ])
            print("======== Max scores updated for {}: {}".format(
                self.domain, scores_diff_repr
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
        self.update_all_priorities(self.compute_priority)

    def pop(self):
        req = super().pop()
        self._print_req(req)
        return req

    def pop_random(self, n_attempts=10):
        req = super().pop_random(n_attempts)
        self._print_req(req)
        return req

    def _print_req(self, req):
        if req:
            scores = get_request_predicted_scores(req, self.G)
            link_tp = '?'
            scores_repr = None
            if scores:
                # scores = {k: v for k, v in scores.items()}
                rewards = dict_subtract(scores, self.max_observed_scores)
                link_tp = sorted(rewards.items(), key=lambda kv: -kv[1])[0][0]
                scores_repr = {k: int(v*100) for k,v in scores.items()}
            print(req.priority, scores_repr or scores, req.url, link_tp.upper())

    def __repr__(self):
        return "DomainRequestQueue({}; #requests={}, max priority={})".format(
            self.domain, len(self), self.max_priority()
        )


class BalancedPriorityQueue:
    """ This queue samples other queues randomly, based on their weights """
    def __init__(self, queue_factory, eps=0.0):
        self.queues = {}  # domain -> queue
        self.eps = eps
        self.queue_factory = queue_factory

        # self.gc_task = LoopingCall(self._gc)
        # self.gc_task.start(60, now=False)

    # def _gc(self):
    #     pass

    def push(self, request):
        domain = request.meta.get('domain')
        if domain not in self.queues:
            self.queues[domain] = self.queue_factory(domain)
        self.queues[domain].push(request)

    def pop(self):
        domains = list(self.queues.keys())
        if not domains:
            return

        random_policy = self.eps and random.random() < self.eps
        if random_policy:
            print("ε", end=' ')

        if random_policy:
            queue = self.queues[random.choice(domains)]
        else:
            weights = [self.queues[domain].max_priority() for domain in domains]
            p = softmax(weights, t=FLOAT_PRIORITY_MULTIPLIER)
            queue = self.queues[np.random.choice(domains, p=p)]
        # print(queue, dict(zip(domains, p)))
        req = queue.pop_random() if random_policy else queue.pop()
        return req

    def update_observed_scores(self, response, observed_scores):
        domain = get_response_domain(response)
        if domain not in self.queues:
            return
        self.queues[domain].update_observed_scores(observed_scores)

    def iter_active_requests(self):
        """ Return an iterator over all requests in a queue """
        for q in self.queues.values():
            yield from q.iter_requests()

    def iter_active_node_ids(self):
        """ Return an iterator over node ids of all queued requests """
        for req in self.iter_active_requests():
            node_id = req.meta.get('node_id')
            if node_id:
                yield node_id

    def recalculate_priorities(self):
        for q in self.queues.values():
            q.recalculate_priorities()

    def __len__(self):
        return sum(len(q) for q in self.queues.values())


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
        if hasattr(spider, 'G'):
            # hack hack hack: if a spider uses crawl graph
            # it is assumed to want BalancedPriorityQueue
            def new_queue(domain):
                return DomainFormFinderRequestsQueue(
                    domain=domain,
                    form_types=available_form_types(),
                    G=spider.G,
                    zeroone_loss=spider.reward_zeroone_loss
                )
            self.queue = BalancedPriorityQueue(
                queue_factory=new_queue,
                eps=spider.epsilon,
            )
        else:
            self.queue = RequestsPriorityQueue(fifo=True)
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
