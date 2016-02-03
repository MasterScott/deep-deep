# -*- coding: utf-8 -*-
import heapq
import itertools

from scrapy.utils.misc import load_object


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
        entry = (-request.priority, count, request)
        heapq.heappush(self.entries, entry)

    def pop(self):
        if not self.entries:
            return None
            # raise KeyError("queue is empty")
        priority, count, request = heapq.heappop(self.entries)
        return request

    @classmethod
    def change_priority(cls, entry, new_priority):
        """ Change priority of an existing entry """
        entry[0] = -new_priority
        entry[2].priority = new_priority

    def heapify(self):
        heapq.heapify(self.entries)

    def __len__(self):
        return len(self.entries)


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
        self.queue = RequestsPriorityQueue()
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
