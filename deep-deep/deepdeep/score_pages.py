# -*- coding: utf-8 -*-
import formasaurus
from deepdeep.utils import dict_aggregate_max


def forms_info(response):
    """ Return a list of form classification results """
    res = formasaurus.extract_forms(response.text, proba=True,
                                    threshold=0, fields=True)
    return [info for form, info in res]


def max_scores(page_forms_info):
    """ Return aggregate form scores for a page """
    return dict_aggregate_max(*[f['form'] for f in page_forms_info])


def response_max_scores(response):
    """ Return aggregate form scores for a page """
    return max_scores(forms_info(response))
