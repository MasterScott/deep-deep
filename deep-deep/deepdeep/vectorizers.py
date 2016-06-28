# -*- coding: utf-8 -*-
from __future__ import absolute_import
from typing import Dict

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import make_union
from sklearn.preprocessing import FunctionTransformer
from formasaurus.text import normalize

from deepdeep.utils import url_path_query, html2text, canonicalize_url


def LinkVectorizer(use_url: bool=False,
                   use_full_url: bool=False,
                   use_same_domain: bool=True
                   ):
    """
    Vectorizer for converting link dicts to feature vectors.
    """
    if use_url and use_full_url:
        raise ValueError("``use_url`` and ``use_full_url`` can't be both True")

    vectorizers = []

    text_vec = HashingVectorizer(
        preprocessor=_link_inside_text,
        n_features=1024*1024,
        binary=True,
        norm='l2',
        # ngram_range=(1, 2),
        analyzer='char',
        ngram_range=(3, 5),
    )
    vectorizers.append(text_vec)

    if use_same_domain:
        same_domain = FunctionTransformer(_same_domain_feature, validate=False)
        vectorizers.append(same_domain)

    if use_url or use_full_url:
        preprocessor = _clean_url if use_url else _clean_url_keep_domain
        url_vec = HashingVectorizer(
            preprocessor=preprocessor,
            n_features=1024*1024,
            binary=True,
            analyzer='char',
            ngram_range=(4, 5),
        )
        vectorizers.append(url_vec)

    return make_union(*vectorizers)


def PageVectorizer():
    """ Vectorizer for converting page HTML content to feature vectors """
    text_vec = HashingVectorizer(
        preprocessor=html2text,
        n_features=1024*1024,
        binary=False,
        ngram_range=(1, 1),
    )
    return text_vec


def _link_inside_text(link: Dict):
    text = link.get('inside_text', '')
    title = link.get('attrs', {}).get('title', '')
    return normalize(text + ' ' + title)


def _clean_url(link: Dict) -> str:
    return url_path_query(_clean_url_keep_domain(link))


def _clean_url_keep_domain(link: Dict) -> str:
    return canonicalize_url(link.get('url'))


def _same_domain_feature(links):
    return np.asarray([
        link['domain_from'] == link['domain_to'] for link in links
    ]).reshape((-1, 1))
