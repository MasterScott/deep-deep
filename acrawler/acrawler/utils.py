# -*- coding: utf-8 -*-
import os
import itertools
from urllib.parse import unquote_plus
from urllib.parse import urlsplit

import numpy as np
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
