import re
from typing import Iterable, List, Tuple
from weakref import WeakKeyDictionary

import html_text  # type: ignore
import pybloom_live  # type: ignore
from scrapy.http.response.text import TextResponse  # type: ignore

from deepdeep.goals import BaseGoal
from .single_domain import SingleDomainSpider, AutopagerBaseline


class UniqueContentGoal(BaseGoal):
    def __init__(self):
        self.crawled = pybloom_live.ScalableBloomFilter()
        self._cache = WeakKeyDictionary()

    def get_reward(self, response: TextResponse) -> float:
        if response not in self._cache:
            if hasattr(response, 'text'):
                new_ngrams = 0
                for ngram in set(html_ngrams(response.text)):
                    new_ngrams += ngram not in self.crawled
                    self.crawled.add(ngram)
                score = new_ngrams / 100.   # arbitrary scaling
            else:
                score = 0.
            self._cache[response] = score
        return self._cache[response]

    def response_observed(self, response: TextResponse):
        pass


def html_ngrams(html: str, n=4) -> Iterable[str]:
    text = html_text.extract_text(html)
    tokens = tokenize(text)
    for ngram in ngrams(tokens, n=n):
        yield ' '.join(ngram)


def tokenize(text: str) -> List[str]:
    return re.findall('\w+', text)


def ngrams(tokens: List[str], n: int) -> Iterable[List[str]]:
    """ All ngrams from 1 to n.
    
    >>> list(ngrams('abcd', 3))
    ['a', 'ab', 'abc', 'b', 'bc', 'bcd', 'c', 'cd', 'd']
    """
    for idx in range(0, len(tokens)):
        for m in range(1, n + 1):
            if idx + m <= len(tokens):
                yield tokens[idx: idx + m]


class UniqueContentSpider(SingleDomainSpider):
    name = 'unique'

    def get_goal(self):
        return UniqueContentGoal()


class UniqueAutopagerBaseline(SingleDomainSpider, AutopagerBaseline):
    name = 'unique_autopager'
