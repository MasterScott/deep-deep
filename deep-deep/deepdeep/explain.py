from typing import Dict, List, Tuple, Union

from scrapy.http.response.text import TextResponse
from eli5.formatters import format_as_html
from eli5.sklearn import explain_prediction
from eli5.sklearn.text import get_weighted_spans
from eli5.sklearn.unhashing import InvertableHashingVectorizer
from eli5.sklearn.utils import FeatureNames
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer

from deepdeep.links import DictLinkExtractor


def get_feature_names_scales(
        vectorizer: FeatureUnion, links: List[Dict], with_scales: bool=True)\
        -> Union[Tuple[FeatureNames, np.ndarray], FeatureNames]:
    """ Assemble feature names and coef scales (if with_scales is True)
    from individual vectorizers, fitting InvertableHashingVectorizer on given links.
    """
    all_features_names = {}
    coef_scales = []
    n_features = 0
    for name, vec in vectorizer.transformer_list:
        if isinstance(vec, HashingVectorizer):
            ivec = InvertableHashingVectorizer(vec)
            ivec.fit(links)
            vec_name = vec.preprocessor.__name__
            feature_names = ivec.get_feature_names(always_signed=False)
            all_features_names.update(
                (n_features + idx,
                 '{} {}'.format(vec_name, name) if isinstance(name, str)
                 else [{'name': '{} {}'.format(vec_name, n['name']),
                        'sign': n['sign']} for n in name])
                for idx, name in feature_names.feature_names.items())
            if with_scales:
                coef_scales.append(ivec.column_signs_)
            n_features += feature_names.n_features
        elif isinstance(vec, FunctionTransformer):
            all_features_names[n_features] = vec.func.__name__
            n_features += 1
            if with_scales:
                coef_scales.append([1.])
    feature_names = FeatureNames(
        all_features_names, n_features=n_features, unkn_template='FEATURE[%d]')
    if with_scales:
        coef_scale = np.empty([sum(map(len, coef_scales))])
        start_idx = 0
        for arr in coef_scales:
            end_idx = start_idx + len(arr)
            coef_scales[start_idx: end_idx] = arr
            start_idx = end_idx
        return feature_names, coef_scale
    else:
        return feature_names


def links_explanations(clf, vec: FeatureUnion, links: List[Dict]) -> List[Dict]:
    all_expl = []
    feature_names = get_feature_names_scales(vec, links, with_scales=False)
    for link in links:
        expl = explain_prediction(
            clf, link, vec, feature_names=feature_names, top=100)
        target_expl = expl['targets'][0]
        add_weighted_spans(link, target_expl, vec)
        all_expl.append((target_expl['score'], link, expl))
    return all_expl


def add_weighted_spans(link, target_expl, vectorizer):
    ws_combined = {'document': '', 'weighted_spans': [], 'not_found': {}}
    for vec_idx in [0, 2]:
        vec = vectorizer.transformer_list[vec_idx][1]
        vec_name = vec.preprocessor.__name__
        feature_weights = {key: [
                (name if isinstance(name, str) else [
                    {'name': n['name'].split(' ', 1)[1], 'sign': n['sign']}
                    for n in name if n['name'].split()[0] == vec_name],
                 coef)
                for name, coef in target_expl['feature_weights'][key]]
                for key in ['pos', 'neg']
                }
        ws = get_weighted_spans(
            link, vec=vec, feature_weights=feature_weights)
        ws_combined['document'] += ' | '
        s0 = len(ws_combined['document'])
        ws_combined['document'] += ws['document']
        shifted_spans = [
            (feature, [(s0 + s, s0 + e) for s, e in spans], weight)
            for feature, spans, weight in ws['weighted_spans']]
        ws_combined['weighted_spans'].extend(shifted_spans)
        for f, w in ws['not_found'].items():
            ws_combined['not_found'][f] = w
    for f, _, _ in ws_combined['weighted_spans']:
        ws_combined['not_found'].pop(f, None)
    target_expl['weighted_spans'] = ws_combined


#   html_expl = format_as_html(expl, include_styles=False, force_weights=False)
#   html_expl = html_expl.replace('<p>Explained as: linear model</p>', '')
#   # show_html(html_expl)
#   all_expl.append((target_expl['score'], link, expl, html_expl))


def extract_links(le: DictLinkExtractor, response: TextResponse) -> List[Dict]:
    return list(le.iter_link_dicts(
        response=response,
        limit_by_domain=False,
        deduplicate=False,
        deduplicate_local=True,
    ))


def item_links(le: DictLinkExtractor, url: str, raw_content: str) -> List[Dict]:
    return extract_links(le, TextResponse(
        url=url,
        body=raw_content,
        encoding='utf8',
    ))
