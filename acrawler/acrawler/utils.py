# -*- coding: utf-8 -*-
import itertools
from formasaurus.utils import get_domain


def aggregate_max(dicts):
    """
    Aggregate dicts by keeping a maximum value for each key.

    >>> dct1 = {'x': 1, 'z': 2}
    >>> dct2 = {'x': 3, 'y': 5, 'z': 1}
    >>> aggregate_max([dct1, dct2]) == {'x': 3, 'y': 5, 'z': 2}
    True
    """
    res = {}
    for dct in dicts:
        for key, value in dct.items():
            res[key] = max(res.get(key, value), value)
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
