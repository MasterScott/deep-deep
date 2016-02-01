# -*- coding: utf-8 -*-
import io
import re
import csv
import logging
import hashlib
import random
from urllib.parse import urlsplit

import scrapy
from scrapy.linkextractors import LinkExtractor

import formasaurus
from formasaurus import formhash
from formasaurus.utils import get_domain
from scrapy.utils.url import add_http_if_no_scheme, guess_scheme

extract_links = LinkExtractor().extract_links


def get_form_hash(form):
    h = formhash.get_form_hash(form).encode('utf8')
    return hashlib.sha1(h).hexdigest()


class BaseSpider(scrapy.Spider):
    seeds_url = None

    def start_requests(self):
        random.seed(0)

        # don't log DepthMiddleware messages
        # see https://github.com/scrapy/scrapy/issues/1308
        logging.getLogger("scrapy.spidermiddlewares.depth").setLevel(logging.INFO)

        if self.seeds_url is None:
            raise ValueError("Please pass seeds_url to the spider. "
                             "It should be a text file with urls, one per line.")

        seeds_url = guess_scheme(self.seeds_url)

        yield scrapy.Request(seeds_url, self.parse_seeds, dont_filter=True,
                             meta={'dont_obey_robotstxt': True})

    def parse_seeds(self, response):
        for num, url in csv.reader(io.StringIO(response.text)):
            url = add_http_if_no_scheme(url)
            yield scrapy.Request(url, self.parse)



class CrawlAllSpider(BaseSpider):

    shuffle = 1  # follow links in order or randomlu
    heuristic = 1  # prefer registration/account links

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.heuristic = int(self.heuristic)
        self.shuffle = int(self.shuffle)
        self.heuristic_re = re.compile("(regi|join|create|sign|account|user|login)")

    def parse(self, response):
        if not hasattr(response, 'text'):
            # can't decode the response
            return

        res = formasaurus.extract_forms(response.text, proba=True, threshold=0, fields=True)
        for form, info in res:
            info['hash'] = get_form_hash(form)
        res = [info for form, info in res]

        # limit crawl to the first domain
        domain = response.meta.get('domain', get_domain(response.url))

        yield {
            'url': response.url,
            'depth': response.meta['depth'],
            'forms': res,
            'domain': domain,
        }

        urls = [link.url for link in extract_links(response)
                if get_domain(link.url) == domain]

        if self.shuffle:
            random.shuffle(urls)

        for idx, url in enumerate(urls):
            N = 5
            # First N random links get priority=0,
            # next N - priority=-1, next N - priority=-2, etc.
            # This way scheduler will prefer to download
            # pages from many domains.
            priority = - (idx // N)

            if self.heuristic:
                # prefer urls which look like registration/login/account urls
                s = self.heuristic_re.search
                p = urlsplit(url)
                if s(p.path) or s(p.query) or s(p.fragment):
                    priority = 1

            yield scrapy.Request(url, meta={'domain': domain}, priority=priority)


class FrontPageSpider(CrawlAllSpider):
    name = 'frontpages'

    custom_settings = {
        'DEPTH_LIMIT': 1,
        'DEPTH_PRIORITY': 1,
    }


class Depth2Spider(CrawlAllSpider):
    name = 'depth2'

    custom_settings = {
        'DEPTH_LIMIT': 2,
        'DEPTH_PRIORITY': 1,
    }
