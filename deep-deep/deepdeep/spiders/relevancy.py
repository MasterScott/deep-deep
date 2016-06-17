# -*- coding: utf-8 -*-
from __future__ import absolute_import
import math
from pathlib import Path

from scrapy.http import Response
from .qspider import QSpider
from deepdeep.goals import RelevancyGoal
from deepdeep.utils import html2text


class RelevancySpider(QSpider):
    """
    This spider learns how to crawl relevant pages.
    """
    name = 'relevant'
    ALLOWED_ARGUMENTS = QSpider.ALLOWED_ARGUMENTS | {'keywords_file'}
    _ARGS = QSpider._ARGS | {'keywords'}

    stay_in_domain = False
    use_pages = 1

    # a file with keywords
    keywords_file = None
    keywords = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keywords = Path(self.keywords_file).read_text().split()
        self._save_params_json()

    def relevancy(self, response: Response) -> float:
        if not hasattr(response, 'text'):
            return 0.0
        text = html2text(response.text).lower()
        score = sum(1.0 if keyword in text else 0.0 for keyword in self.keywords)
        score = math.log(score + 1, len(self.keywords) / 2)
        return score

    def get_goal(self):
        return RelevancyGoal(self.relevancy)

    def _examples(self):
        return None, None
