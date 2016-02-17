# -*- coding: utf-8 -*-
from formasaurus.utils import get_domain
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from formasaurus.text import normalize

from acrawler.utils import url_path_query


# class PartialFitPipeline(Pipeline):
#     """
#     A Pipeline which supports partial_fit with stateless transformers.
#     """
#
#     def partial_fit(self, X, y=None, **fit_transform_params):
#         """Call all the transforms one after the other to transform the
#         data, then call partial_fit for the final estimator.
#
#         Parameters
#         ----------
#         X : iterable
#             Training data. Must fulfill input requirements of first step of the
#             pipeline.
#         y : iterable, default=None
#             Training targets. Must fulfill label requirements for all steps of
#             the pipeline.
#         """
#
#         # prepare
#         params_steps = {step: {} for step, _ in self.steps}
#         for pname, pval in fit_transform_params.items():
#             step, param = pname.split('__', 1)
#             params_steps[step][param] = pval
#
#         # transform
#         Xt = X
#         for name, transform in self.steps[:-1]:
#             Xt = transform.transform(Xt, y, **params_steps[name])
#
#         # fit
#         fit_params = params_steps[self.steps[-1][0]]
#         self.steps[-1][-1].partial_fit(Xt, y, **fit_params)
#         return self
#
#
#
# def get_model(positive_weight=10.0, use_hashing=True):
#     vec = get_vectorizer(use_hashing=use_hashing)
#     clf = get_classifier(positive_weight=positive_weight)
#     return Pipeline([
#         ('vec', vec),
#         ('clf', clf)
#     ])


def get_classifier(positive_weight):
    return SGDClassifier(
        loss='log',
        penalty='elasticnet',
        alpha=1e-6,
        n_iter=1,
        shuffle=False,
        learning_rate='constant',
        eta0=1e-1,
        average=True,
        class_weight={False: 1.0, True: positive_weight},
    )


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

