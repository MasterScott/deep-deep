# -*- coding: utf-8 -*-
from __future__ import absolute_import
import math
from pathlib import Path
from typing import List

from scrapy.http import Response

from .qspider import QSpider
from deepdeep.goals import RelevancyGoal
from deepdeep.utils import html2text
from formasaurus.text import tokenize


def keywords_relevancy(keywords: List[str], response: Response):
    """
    Relevancy score based on how many keywords from a list are
    in response text.

    Score is transformed using a weird log scale (fixme)
    to *roughly* fit [0,1] interval and to not require all keywords to be
    present for a page to be relevant.
    """
    if not hasattr(response, 'text'):
        return 0.0
    text = html2text(response.text).lower()
    tokens = set(tokenize(text))
    score = sum(1.0 if keyword in tokens else 0.0 for keyword in keywords)
    score = math.log(score + 1, len(keywords) / 2)
    return score


class RelevancySpider(QSpider):
    """
    This spider learns how to crawl relevant pages.
    """
    name = 'relevant'
    ALLOWED_ARGUMENTS = {
        'keywords_file',
        'discovery_bonus',
        'max_requests_per_domain',
        'max_relevant_pages_per_domain',
    } | QSpider.ALLOWED_ARGUMENTS
    _ARGS = QSpider._ARGS | {
        'keywords',
        'discovery_bonus',
        'max_requests_per_domain',
        'max_relevant_pages_per_domain'
    }

    stay_in_domain = False
    use_pages = 1
    balancing_temperature = 0.1
    discovery_bonus = 0.0
    max_requests_per_domain = None
    max_relevant_pages_per_domain = None

    # a file with keywords
    keywords_file = None
    keywords = []

    custom_settings = {
        # copied from QSpider
        # 'DEPTH_LIMIT': 100,
        'DEPTH_PRIORITY': 1,
        'SPIDER_MIDDLEWARES': {
            'deepdeep.spidermiddlewares.CrawlGraphMiddleware': 400,
        },

        # disable OffsiteDownloaderMiddleware
        'DOWNLOADER_MIDDLEWARES': {
           'deepdeep.downloadermiddlewares.OffsiteDownloaderMiddleware': None,
        },
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keywords = Path(self.keywords_file).read_text().split()
        self._save_params_json()

    def relevancy(self, response: Response) -> float:
        return keywords_relevancy(self.keywords, response)

    def get_goal(self):
        self.discovery_bonus = float(self.discovery_bonus)
        if self.max_requests_per_domain is not None:
            self.max_requests_per_domain = int(self.max_requests_per_domain)
        if self.max_relevant_pages_per_domain is not None:
            self.max_relevant_pages_per_domain = int(self.max_relevant_pages_per_domain)
        return RelevancyGoal(
            relevancy=self.relevancy,
            discovery_bonus=self.discovery_bonus,
            max_requests_per_domain=self.max_requests_per_domain,
            max_relevant_pages_per_domain=self.max_relevant_pages_per_domain,
        )

    def _examples(self):
        return None, None
