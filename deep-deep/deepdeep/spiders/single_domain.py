# -*- coding: utf-8 -*-
import abc
from typing import Optional

from scrapy import Request  # type: ignore
from scrapy.dupefilters import RFPDupeFilter  # type: ignore

from .qspider import QSpider


class SingleDomainSpider(QSpider, metaclass=abc.ABCMeta):
    """ QSpider with settings tailored to running for a single domain
    with lower memory usage, in order to make it more practical
    to run this spider for item extraction (as opposed to model training).
    Current default configuration will require about 3-6 GB of memory
    for a typical large website.
    This spider also accepts a seed_url argument to specify a single seed url,
    and has an option to run several copies of the spider at the same time
    (to make model more generalizable between different runs).

    Spider arguments
    ----------------
    seed_url : str
        Set this argument in order to start crawling from a single seed URL
        specified from the command line (if you need multiple seeds,
        specify a path to a file with them via seeds_url).
    n_copies : int
        Number of spider "copies" run at the same time (1 by default).
        This copies have independed request queues and cookies, but share
        the same model. This option makes sense when your goal is to train
        a model tha will later be used elsewhere: running several copies reduces
        the chance that the model will learn features that change from run
        to run (e.g. session ids in URLs or depending on a particular order of
        traversal), so the model should be more general.

    It also accepts all arguments accepted by QSpider and BaseSpider.
    """
    use_urls = True
    use_link_text = 1
    use_page_urls = 1
    use_same_domain = 0  # not supported by eli5 yet, and we don't need it
    clf_penalty = 'l1'
    clf_alpha = 0.0001
    balancing_temperature = 5.0  # high to make all simultaneous runs equal
    replay_sample_size = 50
    replay_maxsize = 5000  # single site needs lower replay
    replay_maxlinks = 500000  # some sites can have lots of links per page
    domain_queue_maxsize = 500000

    # single seed url
    seed_url = None  # type: Optional[str]
    # number of simultaneous runs
    n_copies = 1

    _ARGS = {'seed_url', 'n_copies'} | QSpider._ARGS
    ALLOWED_ARGUMENTS = _ARGS | QSpider.ALLOWED_ARGUMENTS

    custom_settings = dict(
        DUPEFILTER_CLASS='deepdeep.spiders.qspider.RunAwareDupeFilter',
        **QSpider.custom_settings)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed_url = self.seed_url
        self.n_copies = int(self.n_copies)

    def start_requests(self):
        if self.seeds_url is None:
            if self.seed_url is None:
                raise ValueError('Pass seeds_url or seed_url')
            yield from self._start_requests([self.seed_url])
        else:
            yield from super().start_requests()

    # Allow running several simultaneous independent spiders on the same domain
    # which still share the model, so it is more general.

    def _start_requests(self, urls):
        for orig_req in super()._start_requests(urls):
            for idx in range(self.n_copies):
                req = orig_req.copy()
                set_run_id(req, 'run-{}'.format(idx))
                yield req

    def _links_to_requests(self, response, *args, **kwargs):
        run_id = response.request.meta['run_id']
        for req in super()._links_to_requests(response, *args, **kwargs):
            set_run_id(req, run_id)
            yield req


class RunAwareDupeFilter(RFPDupeFilter):
    def request_fingerprint(self, request):
        fp = super().request_fingerprint(request)
        return '{}-{}'.format(request.meta.get('run_id'), fp)


def set_run_id(request: Request, run_id: str):
    for key in ['run_id', 'cookiejar', 'scheduler_slot']:
        request.meta[key] = run_id
