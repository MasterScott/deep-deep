# -*- coding: utf-8 -*-
from __future__ import absolute_import
from typing import Dict

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import make_union
from formasaurus.text import normalize
from scrapy.utils.url import canonicalize_url

from deepdeep.utils import url_path_query, html2text


def LinkVectorizer(use_url: bool=False):
    """
    Vectorizer for converting link dicts to feature vectors.
    """
    text_vec = HashingVectorizer(
        preprocessor=_link_inside_text,
        n_features=1024*1024,
        binary=True,
        norm='l2',
        # ngram_range=(1, 2),
        analyzer='char',
        ngram_range=(3, 5),
    )
    if not use_url:
        return text_vec

    url_vec = HashingVectorizer(
        preprocessor=_clean_url,
        n_features=1024*1024,
        binary=True,
        analyzer='char',
        ngram_range=(4,5),
    )
    return make_union(text_vec, url_vec)


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


def _clean_url(link: Dict):
    return url_path_query(canonicalize_url(link.get('url')))
