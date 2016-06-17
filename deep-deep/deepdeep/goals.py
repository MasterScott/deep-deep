# -*- coding: utf-8 -*-
"""
Crawl objectives
================

Crawl objective (goal) classes define how is reward computed.
"""
from __future__ import absolute_import
import abc
from weakref import WeakKeyDictionary

from scrapy.http.response.text import TextResponse
from scrapy.http import Response

from deepdeep.score_pages import response_max_scores
from deepdeep.utils import get_response_domain, MaxScores


class BaseGoal(metaclass=abc.ABCMeta):
    """
    Abstract base class for crawling objectives.
    """

    @abc.abstractmethod
    def get_reward(self, response: Response) -> float:
        """
        Return a reward for a response.
        This method shouldn't update internal goal state;
        implement :meth:`response_observed` method for that.
        """
        pass

    @abc.abstractmethod
    def response_observed(self, response: TextResponse) -> None:
        """
        Update internal state with the received response.
        This method is called after all :meth:`get_reward` calls.
        """
        pass

    def is_acheived_for(self, domain: str) -> bool:
        """
        This method should return True if spider should stop
        processing the website.
        """
        return False

    def debug_print(self) -> None:
        """ Override this method to print debug information during the crawl """
        pass


class FormasaurusGoal(BaseGoal):
    """
    The goal is to find a HTML form of a given type on each website.
    When the form is found, crawling is stopped for a domain.

    ``"password/login recovery"`` forms provide a nice testbed for
    crawling algorithms because a link to the password recovery page is usually
    present on a login page, but not on other website pages. So in order to
    find these forms efficiently crawler must learn to prioritize 'login'
    links, not only 'password recovery' links.

    Parameters
    ----------

    formtype : str
        Form type to look for. Allowed values:

        * "search"
        * "login"
        * "registration"
        * "password/login recovery"
        * "contact/comment"
        * "join mailing list"
        * "order/add to cart"
        * "other"

    threshold : float
         Probability threshold required to consider the goal acheived
         for a domain (default: 0.7).
    """
    def __init__(self, formtype: str, threshold: float=0.7) -> None:
        self.formtype = formtype
        self.threshold = threshold
        self._cache = WeakKeyDictionary()  # type: WeakKeyDictionary
        self._domain_scores = MaxScores()  # domain -> max score

    def get_reward(self, response: TextResponse) -> float:
        if response not in self._cache:
            if hasattr(response, 'text'):
                scores = response_max_scores(response)
                score = scores.get(self.formtype, 0.0)
                # score = score if score > 0.5 else 0
            else:
                score = 0.0
            self._cache[response] = score
        return self._cache[response]

    def response_observed(self, response: TextResponse) -> None:
        reward = self.get_reward(response)
        domain = get_response_domain(response)
        self._domain_scores.update(domain, reward)

    def is_acheived_for(self, domain: str) -> bool:
        score = self._domain_scores[domain]
        should_close = score > self.threshold
        if should_close:
            print("Domain {} is going to be closed; score={:0.4f}.".format(
                domain, score))
        return should_close

    def debug_print(self) -> None:
        print("Scores: sum={:8.1f}, avg={:0.4f}".format(
            self._domain_scores.sum(), self._domain_scores.avg()
        ))
