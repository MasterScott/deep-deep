# -*- coding: utf-8 -*-
"""
Crawl objectives
================

Crawl objective (goal) classes define how is reward computed.
"""
from __future__ import absolute_import
import abc
from typing import Dict, Set, Callable
from weakref import WeakKeyDictionary
from collections import defaultdict

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


class RelevancyGoal(BaseGoal):
    """
    The goal is two-fold:

    1) find new domains which has relevant information;
    2) find relevant information on a website.

    It is implemented by adding a larger bonus for the first relevant page on
    a website; this should encourage spider to go to new domains.

    Parameters
    ----------

    relevancy : callable
        Function to compute relevancy score for a response. It should
        accept scrapy.http.Response and return a score (float value).
        This score is used as reward.
    discovery_bonus: float
        If this is a first page on this domain with
        ``relevancy(response) >= relevancy_threshold`` then
        `discovery_bonus` is added to the reward. Default value is 10.0.
    relevancy_threshold: float
        Minimum relevancy required to give a discovery bonus.
        See `discovery_bonus`.  Default threshold is 0.7.
    """
    def __init__(self,
                 relevancy: Callable[[Response], float],
                 relevancy_threshold: float = 0.5,
                 discovery_bonus: float = 10.0) -> None:
        self.relevancy = relevancy
        self.relevancy_threshold = relevancy_threshold
        self.discovery_bonus = discovery_bonus
        self.relevant_page_found = defaultdict(lambda: False)  # type: defaultdict

    def get_reward(self, response: Response) -> float:
        domain = get_response_domain(response)
        score = self.relevancy(response)
        if score >= self.relevancy_threshold:
            if not self.relevant_page_found[domain]:
                score += self.discovery_bonus
        return score

    def response_observed(self, response: TextResponse) -> None:
        if self.relevancy(response) < self.relevancy_threshold:
            return
        domain = get_response_domain(response)
        self.relevant_page_found[domain] = True


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
