# -*- coding: utf-8 -*-
from __future__ import absolute_import

import itertools
import networkx as nx

import scrapy
from scrapy import signals
from scrapy.utils.request import request_fingerprint
from scrapy.dupefilter import RFPDupeFilter


class BaseExtension:
    def __init__(self, crawler):
        self.crawler = crawler
        self.init()

    def init(self):
        pass

    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler)


class CrawlGraphMiddleware(BaseExtension):
    """
    This spider middleware keeps track of crawl graph.
    The graph is accessible from spider as ``spider.G`` attribute.

    Enable this middleware in settings::

        SPIDER_MIDDLEWARES = {
            'deepdeep.middleware.CrawlGraphMiddleware': 400,
        }

    """
    def init(self):
        # fixme: it should be in spider state
        self.crawler.spider.G = self.G = nx.DiGraph(name='Crawl Graph')
        self.node_ids = itertools.count()
        self.crawler.signals.connect(self.on_spider_closed,
                                     signals.spider_closed)
        self.dupefilter = RFPDupeFilter()  # HACKHACKHACK

    def on_spider_closed(self):
        nx.write_gpickle(self.G, "graph.pickle")

    def process_spider_input(self, response, spider):
        """
        Assign response.node_id attribute, make sure a node exists
        in a graph and update the node with received information.
        """
        if 'node_id' not in response.meta:
            # seed requests don't have node_id yet
            response.meta['node_id'] = next(self.node_ids)

        node_id = response.meta['node_id']
        data = dict(
            url=response.url,
            visited=True,
            ok=self._response_ok(response),
        )
        spider.G.add_node(node_id, data)
        print("VISITED NODE", node_id, data)

    def process_spider_output(self, response, result, spider):
        for request in result:
            if isinstance(request, scrapy.Request):
                ok = self._process_outgoing_request(response, request, spider)
                if not ok:
                    continue
            yield request

    def _process_outgoing_request(self, response, request, spider):
        """
        Create new nodes and edges for outgoing requests.
        Data can be attached to nodes and edges using
        ``request.meta['node_data']`` and ``request.meta['edge_data']``
        dicts; these keys are then removed by this middleware.
        """
        if self.dupefilter.request_seen(request):
            return False

        this_node_id = response.meta.get('node_id')
        new_node_id = next(self.node_ids)
        request.meta['node_id'] = new_node_id

        node_data = request.meta.pop('node_data', {})
        node_data.update(
            url=request.url,
            original_url=request.url,
            visited=False,
            ok=None,
        )
        edge_data = request.meta.pop('edge_data', {})
        spider.G.add_node(new_node_id, node_data)
        spider.G.add_edge(this_node_id, new_node_id, edge_data)
        print("CREATED NODE", this_node_id, "->", new_node_id, node_data)
        return True

    def _response_ok(self, response):
        return response.status == 200 and hasattr(response, 'text')
