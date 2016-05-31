# -*- coding: utf-8 -*-
import csv
import io
import logging
import random

import scrapy
from scrapy.utils.url import canonicalize_url
from scrapy.exceptions import CloseSpider
from scrapy.utils.response import get_base_url
from scrapy.utils.url import guess_scheme, add_http_if_no_scheme
from formasaurus.utils import get_domain

from deepdeep.links import extract_link_dicts
from deepdeep.middlewares import offdomain_request_dropped


class BaseSpider(scrapy.Spider):
    """
    Base spider class witho common code.

    Among other things it parses a file at ``seeds_url`` (URL per line)
    and calls parse for each URL.
    """

    seeds_url = None  # set it in command line
    random_seed = 0
    response_count = 0

    # if you're using command-line arguments override this set in a spider
    # like this:
    # ALLOWED_ARGUMENTS = {'my_new_argument'} | BaseSpider.ALLOWED_ARGUMENTS
    ALLOWED_ARGUMENTS = {
        'seeds_url',
    }

    def __init__(self, *args, **kwargs):
        self._validate_arguments(kwargs)
        super().__init__(*args, **kwargs)

    def _validate_arguments(self, kwargs):
        for k in kwargs:
            if k not in self.ALLOWED_ARGUMENTS:
                raise ValueError(
                    "Unsupported argument: %s. Supported arguments: %r" % (
                        k, self.ALLOWED_ARGUMENTS)
                )

    def start_requests(self):
        self.seen_urls = set()

        # crawer can randomize links to select; make crawl deterministic
        # FIXME: it doesn't make crawl deterministic because
        # scrapy is async and pages can be crawled in different order
        random.seed(int(self.random_seed))

        # don't log DepthMiddleware messages
        # see https://github.com/scrapy/scrapy/issues/1308
        logging.getLogger("scrapy.spidermiddlewares.depth").setLevel(logging.INFO)

        # increase response count on filtered out requests
        self.crawler.signals.connect(self.on_offdomain_request_dropped,
                                     offdomain_request_dropped)

        if self.seeds_url is None:
            raise ValueError("Please pass seeds_url to the spider. It should "
                             "be a text file with urls, one per line.")

        seeds_url = guess_scheme(self.seeds_url)

        yield scrapy.Request(seeds_url, self._parse_seeds, dont_filter=True,
                             meta={'dont_obey_robotstxt': True})

    def _parse_seeds(self, response):
        for url, in csv.reader(io.StringIO(response.text)):
            if url == 'url':
                continue  # optional header
            url = add_http_if_no_scheme(url)
            yield scrapy.Request(url, self.parse)

    def increase_response_count(self):
        """
        Call this method to increase response count and close spider
        if it is over a limit.

        This provides a more flexible alternative to default
        CloseSpider extension.
        """
        self.response_count += 1
        max_items = self.crawler.settings.getint('CLOSESPIDER_ITEMCOUNT',
                                                 float('inf'))
        if self.response_count >= max_items:
            raise CloseSpider("item_count")

    def iter_link_dicts(self, response, domain=None):
        """
        Extract links from the response.
        """
        base_url = get_base_url(response)
        for link in extract_link_dicts(response.selector, base_url):
            url = link['url']

            # only follow in-domain URLs
            if domain is not None and get_domain(url) != domain:
                continue

            # Filter out duplicate URLs.
            # Requests are also filtered out in Scheduler by dupefilter.
            # Here we filter them to avoid creating unnecessary nodes
            # and edges.
            canonical = canonicalize_url(url)
            if canonical in self.seen_urls:
                continue
            self.seen_urls.add(canonical)

            yield link

    def on_offdomain_request_dropped(self, request):
        self.increase_response_count()
