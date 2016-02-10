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
import os
import time
import random
import collections
import datetime

from twisted.internet.task import LoopingCall
import networkx as nx
import scrapy
from scrapy.exceptions import CloseSpider
from scrapy.utils.response import get_base_url
from sklearn.externals import joblib
from formasaurus.utils import get_domain

from acrawler.spiders.base import BaseSpider
from acrawler.utils import (
    get_response_domain,
    set_request_domain,
    decreasing_priority_iter,
    ensure_folder_exists,
)
from acrawler.links import extract_link_dicts
from acrawler import score_links
from acrawler.score_pages import page_scores, available_form_types


class MaxScores:
    """
    >>> s = MaxScores(['x', 'y'])
    >>> s.update("foo", {"x": 0.1, "y": 0.3})
    >>> s.update("foo", {"x": 0.01, "y": 0.4})
    >>> s.update("bar", {"x": 0.8})
    >>> s.sum() == {'x': 0.9, 'y': 0.4}
    True
    >>> s.avg() == {'x': 0.45, 'y': 0.2}
    True
    """
    def __init__(self, classes):
        self.classes = classes
        self._zero_scores = {form_type: 0.0 for form_type in self.classes}
        self.scores = collections.defaultdict(lambda: self._zero_scores.copy())

    def update(self, domain, scores):
        cur_scores = self.scores[domain]
        for k, v in scores.items():
            cur_scores[k] = max(cur_scores[k], v)

    def sum(self):
        return {
            k: sum(v[k] for v in self.scores.values())
            for k in self.classes
        }

    def avg(self):
        if not self.scores:
            return self._zero_scores.copy()
        return {k: v/len(self.scores) for k, v in self.sum().items()}

    def __len__(self):
        return len(self.scores)


class AdaptiveSpider(BaseSpider):
    name = 'adaptive'
    custom_settings = {
        'DEPTH_LIMIT': 3,
        'DEPTH_PRIORITY': 1,
        # 'CONCURRENT_REQUESTS':
    }

    crawl_id = str(datetime.datetime.now())

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.G = nx.DiGraph(name='Crawl Graph')
        self.node_ids = itertools.count()
        self.seen_urls = set()
        self.response_count = 0
        self.domain_scores = MaxScores(available_form_types())

        self.log_task = LoopingCall(self.print_stats)
        self.log_task.start(10, now=False)
        self.checkpoint_task = LoopingCall(self.checkpoint)
        self.checkpoint_task.start(60*10, now=False)

        self.link_vectorizer = score_links.get_vectorizer(use_hashing=True)
        self.link_classifiers = {
            # FIXME: hardcoded 10.0 constant for all form types
            form_cls: score_links.get_classifier(positive_weight=10.0)
            for form_cls in available_form_types()
        }
        ensure_folder_exists(self._data_path(''))
        self.logger.info("Crawl {} started".format(self.crawl_id))

    def parse(self, response):
        self.response_count += 1
        max_items = self.crawler.settings.getint('CLOSESPIDER_ITEMCOUNT') or float('inf')
        if self.response_count >= max_items:
            raise CloseSpider("item_count")

        node_id = self.update_response_node(response)

        if not self.G.node[node_id]['ok']:
            return  # don't send requests from failed responses

        self.update_domain_scores(response, node_id)
        self.update_classifiers(node_id)

        yield from self.generate_out_nodes(response, node_id)

        # TODO:
        # self.update_classifiers_bootstrapped(node_id)

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
        observed_scores = None
        ok = response.status == 200 and hasattr(response, 'text')
        if ok:
            observed_scores = page_scores(response)

        self.G.add_node(
            node_id,
            url=response.url,
            visited=True,
            ok=ok,
            scores=observed_scores,
            response_id=self.response_count,
        )
        return node_id

    def update_domain_scores(self, response, node_id):
        domain = get_response_domain(response)
        scores = self.G.node[node_id]['scores']
        if not scores:
            return
        self.domain_scores.update(domain, scores)

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

        link_scores = self.get_link_scores(links)

        for priority, link, scores in zip(decreasing_priority_iter(), links, link_scores):
            url = link['url']

            # generate nodes and edges
            node_id = next(self.node_ids)
            self.G.add_node(
                node_id,
                url=url,
                visited=False,
                ok=None,
                scores=None,
                response_id=None,
                predicted_scores=scores,
            )
            self.G.add_edge(this_node_id, node_id, link=link)

            # generate Scrapy request
            request = scrapy.Request(url, meta={
                'handle_httpstatus_list': [403, 404, 500],
                'node_id': node_id,
            }, priority=priority)
            set_request_domain(request, domain)
            yield request

    def update_classifiers(self, node_id):
        """ Update classifiers based on information received at node_id """
        node = self.G.node[node_id]
        assert node['visited']

        # We got scores for this node_id; it means we can use incoming links
        # as training data.
        X_raw = []
        for prev_id in self.G.predecessors_iter(node_id):
            link_dict = self.G.edge[prev_id][node_id]['link']
            X_raw.append(link_dict)

        if not X_raw:
            return

        X = self.link_vectorizer.transform(X_raw)

        for form_type, clf in self.link_classifiers.items():
            y = [node['scores'].get(form_type, 0.0) >= 0.5] * len(X_raw)
            clf.partial_fit(X, y, classes=[False, True])

    def update_classifiers_bootstrapped(self, node_id):
        """ Update classifiers based on outgoing link scores """
        # TODO
        raise NotImplementedError()

    def get_link_scores(self, links):
        if not links:
            return []
        X = self.link_vectorizer.transform(links)
        scores = [{} for _ in links]
        for form_type, clf in self.link_classifiers.items():
            if clf.coef_ is None:
                continue  # not fitted yet
            probs = clf.predict_proba(X)[..., 1]
            for prob, score_dict in zip(probs, scores):
                score_dict[form_type] = prob
        return scores

    def iter_link_dicts(self, response, limit_domain):
        base_url = get_base_url(response)
        for link in extract_link_dicts(response.selector, base_url):
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

    def print_stats(self):
        msg = "Crawl graph: {} nodes ({} visited), {} edges, {} domains".format(
            self.G.number_of_nodes(),
            self.response_count,
            self.G.number_of_edges(),
            len(self.domain_scores)
        )
        self.logger.info(msg)

        scores_sum = sorted(self.domain_scores.sum().items())
        scores_avg = sorted(self.domain_scores.avg().items())
        reward_lines = [
            "{:8.1f}   {:0.4f}   {}".format(tot, avg, k)
            for ((k, tot), (k, avg)) in zip(scores_sum, scores_avg)
        ]
        msg = '\n'.join(reward_lines)
        self.logger.info("Reward (total / average): \n{}".format(msg))

    def checkpoint(self):
        ts = int(time.time())
        graph_filename = 'crawl-{}.pickle.gz'.format(ts)
        clf_filename = 'classifiers-{}.joblib'.format(ts)
        self.save_crawl_graph(graph_filename)
        self.save_classifiers(clf_filename)

    def save_crawl_graph(self, path):
        self.logger.info("Saving crawl graph...")
        nx.write_gpickle(self.G, self._data_path(path))
        self.logger.info("Crawl graph saved")

    def save_classifiers(self, path):
        self.logger.info("Saving classifiers...")
        pipe = {
            'vec': self.link_vectorizer,
            'clf': self.link_classifiers,
        }
        joblib.dump(pipe, self._data_path(path), compress=3)
        self.logger.info("Classifiers saved")

    def _data_path(self, path):
        return os.path.join('checkpoints', self.crawl_id, path)

    def closed(self, reason):
        """ Save crawl graph to a file when spider is closed """
        for task in [self.log_task, self.checkpoint_task]:
            if task.running:
                task.stop()
        self.save_classifiers('classifiers.joblib')
        self.save_crawl_graph('crawl.pickle.gz')
