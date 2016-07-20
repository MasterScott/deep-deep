# -*- coding: utf-8 -*-
from __future__ import absolute_import
import math
from pathlib import Path
from typing import List, Optional

from scrapy.http import Response
from formasaurus.text import tokenize, token_ngrams

from .qspider import QSpider
from deepdeep.goals import RelevancyGoal
from deepdeep.utils import html2text


class RelevancySpider(QSpider):
    """
    This spider learns how to crawl relevant pages.
    """
    name = 'relevant'
    ALLOWED_ARGUMENTS = {
        'keywords_file',
        'max_requests_per_domain',
        'max_relevant_pages_per_domain',
    } | QSpider.ALLOWED_ARGUMENTS
    _ARGS = QSpider._ARGS | {
        'pos_keywords',
        'neg_keywords',
        'max_requests_per_domain',
        'max_relevant_pages_per_domain'
    }

    stay_in_domain = False
    use_pages = 1
    balancing_temperature = 0.1
    max_requests_per_domain = None  # type: Optional[int]
    max_relevant_pages_per_domain = None  # type: Optional[int]
    replay_sample_size = 50
    replay_maxsize = 10000  # increase it if use_pages is 0

    # a file with keywords
    keywords_file = None   # type: str
    pos_keywords = []      # type: List[str]
    neg_keywords = []      # type: List[str]

    custom_settings = {
        # copied from QSpider
        # 'DEPTH_LIMIT': 100,
        'DEPTH_PRIORITY': 1,

        # disable OffsiteDownloaderMiddleware
        'DOWNLOADER_MIDDLEWARES': {
           'deepdeep.downloadermiddlewares.OffsiteDownloaderMiddleware': None,
        },
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        keywords = Path(self.keywords_file).read_text().splitlines()
        self.pos_keywords = [k for k in keywords if not k.startswith('-')]
        self.neg_keywords = [k[1:] for k in keywords if k.startswith('-')]
        self.max_ngram = _max_ngram_length(self.pos_keywords)
        self._save_params_json()

    def relevancy(self, response: Response) -> float:
        return keywords_response_relevancy(response,
                                           pos_keywords=self.pos_keywords,
                                           neg_keywords=self.neg_keywords,
                                           max_ngram=self.max_ngram)

    def get_goal(self):
        if self.max_requests_per_domain is not None:
            self.max_requests_per_domain = int(self.max_requests_per_domain)
        if self.max_relevant_pages_per_domain is not None:
            self.max_relevant_pages_per_domain = int(self.max_relevant_pages_per_domain)
        return RelevancyGoal(
            relevancy=self.relevancy,
            max_requests_per_domain=self.max_requests_per_domain,
            max_relevant_pages_per_domain=self.max_relevant_pages_per_domain,
        )


def keywords_response_relevancy(response: Response,
                                pos_keywords: List[str],
                                neg_keywords: List[str],
                                max_ngram=1):
    """
    Relevancy score based on how many keywords from a list are
    in response text.

    Score is transformed using a weird log scale (fixme)
    to *roughly* fit [0,1] interval and to not require all keywords to be
    present for a page to be relevant.
    """
    if not hasattr(response, 'text'):
        return 0.0
    return keyword_relevancy(response.text, pos_keywords, neg_keywords, max_ngram)


def keyword_relevancy(response_html: str,
                      pos_keywords: List[str],
                      neg_keywords: List[str],
                      max_ngram=1):
    text = html2text(response_html).lower()
    tokens = tokenize(text)
    tokens = set(token_ngrams(tokens, 1, max_ngram))

    def _score(keywords: List[str]) -> float:
        s = sum(int(k in tokens) for k in keywords)
        return _scale_relevancy(s, keywords)

    pos_score = _score(pos_keywords)
    neg_score = _score(neg_keywords)

    return max(0, pos_score - 0.33 * neg_score)


def _max_ngram_length(keywords: List[str]) -> int:
    """
    >>> _max_ngram_length(["foo"])
    1
    >>> _max_ngram_length(["foo", "foo  bar"])
    2
    >>> _max_ngram_length(["  foo", "foo bar", "foo bar baz "])
    3
    """
    return max(len(keyword.split()) for keyword in keywords)


def _scale_relevancy(score: float, keywords: List) -> float:
    """ Weird log scale to use for keyword occurance count """
    return math.log(score + 1, len(keywords) / 2 + 2)

