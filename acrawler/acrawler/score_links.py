# -*- coding: utf-8 -*-
from formasaurus.utils import get_domain
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from formasaurus.text import normalize

from acrawler.utils import url_path_query


def get_classifier(positive_weight, converge):
    params = dict(
        loss='log',
        penalty='elasticnet',
        alpha=1e-6,
        n_iter=1,
        shuffle=False,
        average=True,
        class_weight={False: 1.0, True: positive_weight},
    )
    if converge:
        params.update(
            learning_rate='optimal',
        )
    else:
        params.update(
            learning_rate='constant',
            eta0=1e-1,
        )

    return SGDClassifier(**params)


def get_vectorizer(use_hashing, use_domain):
    cls = HashingVectorizer if use_hashing else CountVectorizer

    # link url
    vec_url = cls(
        preprocessor=_link_url,
        analyzer='char',
        binary=True,
        ngram_range=(4,4),
    )

    # link text
    vec_text = cls(
        preprocessor=_link_inside_text,
        analyzer='word',
        binary=True,
        ngram_range=(1,2),
    )

    features = [
        ('url', vec_url),
        ('text', vec_text),
    ]

    if use_domain:
        vec_domain = cls(
            preprocessor=_link_domain,
            analyzer='word',
            binary=True,
        )
        features.append(('domain', vec_domain))

    return FeatureUnion(features)


def _link_url(link):
    return url_path_query(link['url'])


def _link_inside_text(link):
    text = link.get('inside_text', '')
    title = link['attrs'].get('title', '')
    return normalize(text + ' ' + title)


def _link_domain(link):
    return get_domain(link['url'])


def _link_before_text(link):
    return normalize(link.get('before_text', ''))

