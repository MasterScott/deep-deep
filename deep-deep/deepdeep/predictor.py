# -*- coding: utf-8 -*-
from __future__ import absolute_import
from typing import List, Tuple

import joblib
import parsel
from w3lib.html import get_base_url

from deepdeep.links import extract_link_dicts
from deepdeep.qlearning import QLearner
from deepdeep.utils import get_domain


class LinkClassifier:
    """
    This class allows to use Q.joblib models saved by QSpider.
    Load it with ``clf = LinkClassifier.load('/path/to/Q.joblib')``,
    then call :meth:`extract_urls` to get all links on a page along
    with their scores.
    """
    def __init__(self, Q, link_vectorizer, page_vectorizer, **kwargs):
        self.Q = Q  # type: QLearner
        self.link_vectorizer = link_vectorizer
        self.page_vectorizer = page_vectorizer
        self.extra = kwargs

    @classmethod
    def load(cls, path):
        model = joblib.load(str(path))
        return cls(**model)

    def extract_urls(self, html: str, url: str) -> List[Tuple[float, str]]:
        """
        Extract all URLs from html, return a list of (score, url) pairs.
        """
        sel = parsel.Selector(html)
        base_url = get_base_url(html[:4096], url)
        links = list(extract_link_dicts(sel, base_url))
        if not links:
            return []

        domain_from = get_domain(url)
        for link in links:
            link['domain_from'] = domain_from
            link['domain_to'] = get_domain(link['url'])

        if self.page_vectorizer:
            page_vec = self.page_vectorizer.transform([html])
        else:
            page_vec = None
        link_matrix = self.link_vectorizer.transform(links)
        AS = self.Q.join_As(link_matrix, page_vec)
        scores = self.Q.predict(AS)

        urls = [link['url'] for link in links]
        return list(zip(scores, urls))
