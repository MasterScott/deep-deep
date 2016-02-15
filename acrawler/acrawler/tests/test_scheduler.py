# -*- coding: utf-8 -*-
import scrapy

from acrawler.scheduler import (
    RequestsPriorityQueue,
    DomainFormFinderRequestsQueue,
)


def test_request_priority_queue():
    q = RequestsPriorityQueue(fifo=True)
    q.push(scrapy.Request('http://example.com/1', priority=1))
    q.push(scrapy.Request('http://example.com/1/1', priority=1))
    q.push(scrapy.Request('http://example.com/-1', priority=-1))
    q.push(scrapy.Request('http://example.com/2', priority=2))
    q.push(scrapy.Request('http://example.com/0', priority=0))

    assert q.get_priority(q.entries[0]) == 2
    assert len(q) == 5

    assert q.pop().url == "http://example.com/2"
    assert len(q) == 4
    assert q.pop().url == "http://example.com/1"
    assert q.pop().url == "http://example.com/1/1"
    assert q.pop().url == "http://example.com/0"
    assert q.pop().url == "http://example.com/-1"
    assert len(q) == 0
