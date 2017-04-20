import importlib
import traceback
from typing import Any, Callable, Iterable, Set, Tuple
from weakref import WeakKeyDictionary

from scrapy import Request  # type: ignore
from scrapy.http.response.text import TextResponse  # type: ignore

from .single_domain import SingleDomainSpider, AutopagerBaseline
from deepdeep.goals import BaseGoal


class ExtractionGoal(BaseGoal):
    def __init__(self,
                 extractor: Callable[[TextResponse], Iterable[Tuple[Any, Any]]],
                 request_penalty: float=1.0,
                 item_callback=None,
                 ) -> None:
        """ The goal is to find the maximum number of unique items by doing
        minimum number of requests.

        Parameters
        ----------
        extractor : callable
            A function that extracts key-item pairs for each item found in
            response. Key is a unique item identifier
            (like item_id or item_type and item_id), and item is extracted
            data.
        request_penalty : float
            Penalty for making a request (default: 1.0).
            Reward is calculated as number of items minus request penalty.
        item_callback : callable
            A function that will be called with response.url, key, item
            for each extracted item.
        """
        self.extractor = extractor
        self.extracted_items = set()  # type: Set[Tuple[str, str]]
        self.request_reward = -request_penalty
        self.item_reward = 1.0
        self.item_callback = item_callback
        self._cache = WeakKeyDictionary()  # type: WeakKeyDictionary

    def get_reward(self, response: TextResponse) -> float:
        if response not in self._cache:
            score = self.request_reward
            run_id = response.meta['run_id']
            try:
                items = list(self.extractor(response))
            except Exception:
                traceback.print_exc()
            else:
                for key, item in items:
                    full_key = (run_id, key)
                    if full_key not in self.extracted_items:
                        self.extracted_items.add(full_key)
                        score += self.item_reward
                    if self.item_callback:
                        self.item_callback(response.url, key, item)
            self._cache[response] = score
        return self._cache[response]

    def response_observed(self, response: TextResponse):
        pass


class ExtractionSpider(SingleDomainSpider):
    """
    This spider learns how to extract data from a single domain.
    It uses ExtractionGoal goal (extracting maximum number of unique items using
    minimal number of requests).

    Spider arguments
    ----------------
    extractor : str
        This required argument specifies the python path to the extractor
        function, and has the form "python.module:function". This function is
        passed as ``extractor`` argument to ``ExtractionGoal``.
    export_items : bool
        Set this option to get extracted items in spider output. The format
        Each unique item returned by the extractor function will produce an item
        with 3 fields: 'url' is the response url,
        'key' is the key returned by the extractor function, and item is item
        returned by the extractor function.

    It also accepts all arguments accepted by QSpider, SingleDomainSpider
    and BaseSpider.
    """
    name = 'extraction'
    export_items = 1
    export_cdr = 0

    _ARGS = ({'extractor', 'export_items'} | SingleDomainSpider._ARGS)
    ALLOWED_ARGUMENTS = _ARGS | SingleDomainSpider.ALLOWED_ARGUMENTS

    def __init__(self, *args, **kwargs):
        """ extractor argument has a "module:function" format
        and specifies where to load the extractor from.
        """
        super().__init__(*args, **kwargs)
        self.extractor = str(self.extractor)
        self.export_items = bool(int(self.export_items))
        self.exported_keys = set()
        self.export_buffer = []

    def get_goal(self):
        try:
            ex_module, ex_function = self.extractor.split(':')
        except (AttributeError, ValueError):
            raise ValueError(
                'Please give extractor argument in "module:function" format')
        ex_module = importlib.import_module(ex_module)
        extractor_fn = getattr(ex_module, ex_function)
        return ExtractionGoal(extractor_fn, item_callback=self.item_callback)

    def item_callback(self, url, key, item):
        if self.export_items and key not in self.exported_keys:
            self.export_buffer.append({'url': url, 'key': key, 'item': item})
            self.exported_keys.add(key)

    def parse(self, response):
        parse_result = super().parse(response)
        self.log_value('Reward/total-items', len(self.exported_keys))
        if self.export_items:
            yield from self.export_buffer
            self.export_buffer = []
            for item_or_link in parse_result:
                if isinstance(item_or_link, Request):
                    yield item_or_link
        else:
            yield from parse_result


class AutopagerExtractionBaseline(ExtractionSpider, AutopagerBaseline):
    name = 'autopager_extraction'
