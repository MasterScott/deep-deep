# -*- coding: utf-8 -*-
import re
import random
from urllib.parse import urlsplit

import scrapy
from scrapy.linkextractors import LinkExtractor
from acrawler.utils import (
    get_response_domain,
    set_request_domain,
)
from formasaurus.utils import get_domain

from acrawler.spiders.base import BaseSpider
from acrawler.score_pages import forms_info
from acrawler.utils import decreasing_priority_iter


class CrawlAllSpider(BaseSpider):
    """
    Spider for crawling experiments.

    It is written as a single spider with arguments (not as multiple spiders)
    in order to share HTTP cache.
    """
    name = 'all'

    shuffle = 1  # follow links in order or randomly
    heuristic = 0  # prefer registration/account links

    custom_settings = {
        'DEPTH_LIMIT': 1,  # override it using -s DEPTH_LIMIT=2
        'DEPTH_PRIORITY': 1,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.heuristic_re = re.compile("(regi|join|create|sign|account|user|login)")
        self.heuristic = int(self.heuristic)
        self.shuffle = int(self.shuffle)
        self.extract_links = LinkExtractor().extract_links

    def parse(self, response):
        self.increase_response_count()

        if not hasattr(response, 'text'):
            # can't decode the response
            return

        res = forms_info(response)

        yield {
            'url': response.url,
            'depth': response.meta['depth'],
            'forms': res,
            'domain': get_response_domain(response),
        }

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
        domain = get_response_domain(response)
        urls = [link.url for link in self.extract_links(response)
                if get_domain(link.url) == domain]

        if shuffle:
            random.shuffle(urls)

        for priority, url in zip(decreasing_priority_iter(), urls):
            if prioritize_re:
                s = prioritize_re.search
                p = urlsplit(url)
                if s(p.path) or s(p.query) or s(p.fragment):
                    priority = 1

            req = scrapy.Request(url, priority=priority)
            set_request_domain(req, domain)
            yield req
