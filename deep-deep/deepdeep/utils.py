# -*- coding: utf-8 -*-
import os
import itertools
import functools
import collections
from urllib.parse import unquote_plus
from urllib.parse import urlsplit

import numpy as np
import time

from formasaurus.utils import get_domain


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


def dict_subtract(d1, d2):
    """
    Subtract values in d2 from values in d1.

    >>> d1 = {'x': 1, 'y': 2, 'z': 3}
    >>> d2 = {'x': 2, 'y': 1, 'w': 3}
    >>> dict_subtract(d1, d2) == {'x': -1, 'y': 1, 'z': 3, 'w': -3}
    True
    """
    res = d1.copy()
    for k, v in d2.items():
        res[k] = res.get(k, 0) - v
    return res


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


def url_path_query(url):
    """
    Return URL path and query, without domain, scheme and fragment:

    >>> url_path_query("http://example.com/foo/bar?k=v&egg=spam#id9")
    '/foo/bar?k=v&egg=spam'
    """
    p = urlsplit(url)
    return unquote_plus(p.path + '?' + p.query).lower()


def ensure_folder_exists(path):
    """ Create folder `path` if necessary """
    os.makedirs(path, exist_ok=True)


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
    >>> s = MaxScores(['x', 'y'])
    >>> s.update("foo", {"x": 0.1, "y": 0.3})
    >>> s.update("foo", {"x": 0.01, "y": 0.4})
    >>> s.update("bar", {"x": 0.8})
    >>> s.sum() == {'x': 0.9, 'y': 0.4}
    True
    >>> s.avg() == {'x': 0.45, 'y': 0.2}
    True
    >>> len(s)
    2
    """
    def __init__(self, classes):
        self.classes = classes
        self._zero_scores = {key: 0.0 for key in self.classes}
        self.scores = collections.defaultdict(lambda: self._zero_scores.copy())

    def update(self, domain, scores):
        cur_scores = self.scores[domain]
        for k, v in scores.items():
            cur_scores[k] = max(cur_scores[k], v)

    def sum(self):
        return {
            k: sum(v[k] for v in self.scores.values())
            for k in self.classes
        }

    def avg(self):
        if not self.scores:
            return self._zero_scores.copy()
        return {k: v/len(self.scores) for k, v in self.sum().items()}

    def __getitem__(self, domain):
        return self.scores[domain]

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
            print("{} took {:0.4f}s".format(func, end-start))
    return wrapper
