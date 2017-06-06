# -*- coding: utf-8 -*-
import logging
import time
import itertools
import functools
import collections
from urllib.parse import unquote_plus, urlsplit

import numpy as np  # type: ignore
from scipy.sparse.csr import csr_matrix  # type: ignore
import tldextract  # type: ignore
from scrapy.utils.url import canonicalize_url as _canonicalize_url  # type: ignore


logger = logging.getLogger(__name__)


def dict_aggregate_max(*dicts):
    """
    Aggregate dicts by keeping a maximum value for each key.

    >>> dct1 = {'x': 1, 'z': 2}
    >>> dct2 = {'x': 3, 'y': 5, 'z': 1}
    >>> dict_aggregate_max(dct1, dct2) == {'x': 3, 'y': 5, 'z': 2}
    True
    """
    res = {}
    for dct in dicts:
        for key, value in dct.items():
            res[key] = max(res.get(key, value), value)
    return res


def get_domain(url: str) -> str:
    """
    >>> get_domain("http://foo.com")
    'foo.com'
    >>> get_domain("http://bar.FOO.com/path")
    'foo.com'
    >>> get_domain("http://foo.onion")
    'foo.onion'
    >>> get_domain("http://bar.foo.onion")
    'foo.onion'
    """
    e = tldextract.extract(url)
    if not e.suffix and e.domain == 'onion':
        subdomain = e.subdomain.rsplit('.', 1)[-1]
        domain = '{}.{}'.format(subdomain, e.domain)
    else:
        domain = e.registered_domain
    return domain.lower()


def get_response_domain(response):
    return response.meta.get('domain', get_domain(response.url))


def set_request_domain(request, domain):
    request.meta['domain'] = domain


def decreasing_priority_iter(N=5):
    # First N random links get priority=0,
    # next N - priority=-1, next N - priority=-2, etc.
    # This way scheduler will prefer to download
    # pages from many domains.
    for idx in itertools.count():
        priority = - (idx // N)
        yield priority


def url_path_query(url: str) -> str:
    """
    Return URL path and query, without domain, scheme and fragment:

    >>> url_path_query("http://example.com/foo/bar?k=v&egg=spam#id9")
    '/foo/bar?k=v&egg=spam'
    """
    p = urlsplit(url)
    return unquote_plus(p.path + '?' + p.query).lower()


def softmax(z, t=1.0):
    """
    Softmax function with temperature.

    >>> softmax(np.zeros(4))
    array([ 0.25,  0.25,  0.25,  0.25])
    >>> softmax([])
    array([], dtype=float64)
    >>> softmax([-2.85, 0.86, 0.28])  # DOCTEST: +ELLIPSES
    array([ 0.015...,  0.631...,  0.353...])
    >>> softmax([-2.85, 0.86, 0.28], t=0.00001)
    array([ 0.,  1.,  0.])
    """
    if not len(z):
        return np.array([])

    z = np.asanyarray(z) / t
    z_exp = np.exp(z - np.max(z))
    return z_exp / z_exp.sum()


class MaxScores:
    """
    >>> s = MaxScores()
    >>> s.update("foo", 0.2)
    >>> s.update("foo", 0.1)
    >>> s.update("bar", 0.5)
    >>> s.update("bar", 0.6)
    >>> s['unknown']
    0
    >>> s['foo']
    0.2
    >>> s['bar']
    0.6
    >>> s.sum()
    0.8
    >>> s.avg()
    0.4
    >>> len(s)
    2
    """
    def __init__(self, default=0):
        self.default = default
        self.scores = collections.defaultdict(lambda: default)

    def update(self, key, value):
        self.scores[key] = max(self.scores[key], value)

    def sum(self):
        return sum(self.scores.values())

    def avg(self):
        if len(self) == 0:
            return 0
        return self.sum() / len(self)

    def __getitem__(self, key):
        if key not in self.scores:
            return self.default
        return self.scores[key]

    def __len__(self):
        return len(self.scores)


def log_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            end = time.time()
            logging.debug("{} took {:0.4f}s".format(func, end-start))
    return wrapper


@functools.lru_cache(maxsize=100000)
def canonicalize_url(url: str) -> str:
    return _canonicalize_url(url)


def csr_nbytes(m: csr_matrix) -> int:
    if m is not None:
        return m.data.nbytes + m.indices.nbytes + m.indptr.nbytes
    else:
        return 0


def chunks(lst, chunk_size: int):
    for idx in range(0, len(lst), chunk_size):
        yield lst[idx: idx + chunk_size]
