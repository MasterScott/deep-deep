# -*- coding: utf-8 -*-
import re
import hashlib
import random
from urllib.parse import urlsplit

import scrapy
from scrapy.linkextractors import LinkExtractor
import formasaurus
from formasaurus import formhash
from formasaurus.utils import get_domain

from .base import BaseSpider


extract_links = LinkExtractor().extract_links


def get_form_hash(form):
    h = formhash.get_form_hash(form).encode('utf8')
    return hashlib.sha1(h).hexdigest()


def forms_info(response):
    """ Return a list of form classification results """
    res = formasaurus.extract_forms(response.text, proba=True,
                                    threshold=0, fields=True)
    for form, info in res:
        info['hash'] = get_form_hash(form)
    return [info for form, info in res]


class CrawlAllSpider(BaseSpider):
    """
    Spider for crawling experiments.

    It is written as a single spider with arguments (not as multiple spiders)
    in order to share HTTP cache.
    """
    name = 'all'

    shuffle = 1  # follow links in order or randomly
    heuristic = 0  # prefer registration/account links
    smart = 0  # enable smart crawling

    custom_settings = {
        'DEPTH_LIMIT': 1,  # override it using -s DEPTH_LIMIT=2
        'DEPTH_PRIORITY': 1,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.heuristic_re = re.compile("(regi|join|create|sign|account|user|login)")
        self.heuristic = int(self.heuristic)
        self.shuffle = int(self.shuffle)
        self.smart = int(self.smart)

    @classmethod
    def get_domain(cls, response):
        return response.meta.get('domain', get_domain(response.url))

    def parse(self, response):
        if not hasattr(response, 'text'):
            # can't decode the response
            return

        res = forms_info(response)

        yield {
            'url': response.url,
            'depth': response.meta['depth'],
            'forms': res,
            'domain': self.get_domain(response),
        }

        if self.smart:
            yield from self.crawl_smart(response)
        else:
            yield from self.crawl_baseline(response,
                shuffle=self.shuffle,
                prioritize_re=None if not self.heuristic else self.heuristic_re
            )

    def crawl_baseline(self, response, shuffle, prioritize_re=None):
        """
        Baseline crawling algoritms.

        When shuffle=True, links are selected at random.
        When prioritize_re is not None, links which URLs follow specified
        regexes are prioritized.
        """

        # limit crawl to the first domain
        domain = self.get_domain(response)
        urls = [link.url for link in extract_links(response)
                if get_domain(link.url) == domain]

        if shuffle:
            random.shuffle(urls)

        for idx, url in enumerate(urls):
            N = 5
            # First N random links get priority=0,
            # next N - priority=-1, next N - priority=-2, etc.
            # This way scheduler will prefer to download
            # pages from many domains.
            priority = - (idx // N)

            if prioritize_re:
                s = prioritize_re.search
                p = urlsplit(url)
                if s(p.path) or s(p.query) or s(p.fragment):
                    priority = 1

            yield scrapy.Request(url, meta={'domain': domain}, priority=priority)

    def crawl_smart(self, response):
        """
        Adaptive crawling algorithm.
        """
        pass
