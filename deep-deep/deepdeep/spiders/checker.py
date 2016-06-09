# -*- coding: utf-8 -*-
from ._base import BaseSpider


class CheckerSpider(BaseSpider):
    """
    Cleanup URL lists using this spider::

        scrapy crawl checker -a seeds_url=../alexa1k.csv -o urls.csv -L INFO

    """
    name = 'checker'

    custom_settings = {
        'RETRY_ENABLED': True,
        'RETRY_TIMES': 1,
        'DOWNLOAD_TIMEOUT': 60,
        'DOWNLOAD_MAXSIZE': 1024*1024*32  # 32MB
    }

    def parse(self, response):
        if not hasattr(response, 'text'):
            return  # not a text response

        if not response.text:
            return  # empty response

        yield {'url': response.url}
