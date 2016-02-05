# -*- coding: utf-8 -*-
"""
Adaptive crawling algorithm.

The first assumption is that all links to the same page are similar
if they are from the same domain. Because crawler works in-domain
it means we don't have to turn off dupefilter, and that there is no
need to handle all incoming links to a page - it is enough to
consider only one. This means instead of a general crawl graph
we're working with a crawl tree.
"""

import itertools
import logging
import random

import networkx as nx
import scrapy

from acrawler.spiders.base import BaseSpider
from acrawler.classifiers import page_scores
from acrawler.utils import (
    get_response_domain,
    set_request_domain,
    decreasing_priority_iter)
from acrawler.links import extract_link_dicts
from formasaurus.utils import get_domain



class AdaptiveSpider(BaseSpider):
    name = 'adaptive'

    custom_settings = {
        'DEPTH_PRIORITY': 1,
        # 'CONCURRENT_REQUESTS':
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.G = nx.DiGraph(name='Crawl Graph')
        self.node_ids = itertools.count()
        self.seen_urls = set()

    def parse(self, response):
        node_id = self.update_response_node(response)

        if not self.G.node[node_id]['ok']:
            return  # don't send requests from failed responses

        yield from self.generate_out_nodes(response, node_id)

    def update_response_node(self, response):
        """
        Update crawl graph with information about the received response.
        Return node_id of the node which corresponds to this response.
        """
        node_id = response.meta.get('node_id')

        # 1. Handle seed responses which don't yet have node_id
        if node_id is None:
            node_id = next(self.node_ids)

        # 2. Update node with observed information
        ok = response.status == 200 and hasattr(response, 'text')
        if ok:
            observed_scores = page_scores(response)
        else:
            observed_scores = None

        self.G.add_node(
            node_id,
            url=response.url,
            visited=True,
            ok=ok,
            scores=observed_scores,
        )
        return node_id

    def generate_out_nodes(self, response, this_node_id):
        """
        Extract links from the response and add nodes and edges to crawl graph.
        Returns an iterator of scrapy.Request objects.
        """

        # Extract in-domain links and their features
        domain = get_response_domain(response)

        # Generate nodes, edges and requests based on link information
        links = list(self.iter_link_dicts(response, domain))
        random.shuffle(links)

        for priority, link in zip(decreasing_priority_iter(), links):
            url = link['url']

            # generate nodes and edges
            node_id = next(self.node_ids)
            self.G.add_node(
                node_id,
                url=url,
                visited=False,
                ok=None,
                scores={},  # TODO: estimate scores
            )
            self.G.add_edge(this_node_id, node_id, link=link)

            # generate Scrapy request
            request = scrapy.Request(url, meta={
                'handle_httpstatus_list': [403, 404, 500],
                'node_id': node_id,
            }, priority=priority)
            set_request_domain(request, domain)
            yield request

    def iter_link_dicts(self, response, limit_domain):
        for link in extract_link_dicts(response):
            url = link['url']

            # only follow in-domain URLs
            if get_domain(url) != limit_domain:
                continue

            # Filter out duplicate URLs.
            # Requests are also filtered out in Scheduler by dupefilter.
            # Here we filter them to avoid creating unnecessary nodes
            # and edges.
            if url in self.seen_urls:
                continue
            self.seen_urls.add(url)

            yield link

    def closed(self, reason):
        """ Save crawl graph to a file when spider is closed """
        self.logger.info("Saving crawl graph...",)
        nx.write_gpickle(self.G, 'crawl.pickle.gz')
        self.logger.info("Crawl graph saved")

