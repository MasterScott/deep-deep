# -*- coding: utf-8 -*-
import hashlib

import formasaurus
from formasaurus import formhash
from deepdeep.utils import dict_aggregate_max


def get_form_hash(form):
    h = formhash.get_form_hash(form).encode('utf8')
    return hashlib.sha1(h).hexdigest()


def forms_info(response):
    """ Return a list of form classification results """
    res = formasaurus.extract_forms(response.text, proba=True,
                                    threshold=0, fields=True)
    for form, info in res:
        info['hash'] = get_form_hash(form)
    return [info for form, info in res]


def max_scores(page_forms_info):
    """ Return aggregate form scores for a page """
    return dict_aggregate_max(*[f['form'] for f in page_forms_info])


def response_max_scores(response):
    """ Return aggregate form scores for a page """
    return max_scores(forms_info(response))
