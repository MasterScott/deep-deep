# -*- coding: utf-8 -*-
import hashlib

import formasaurus
from formasaurus import formhash, classifiers
from acrawler.utils import aggregate_max


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


def page_scores(response):
    """ Return aggregate form scores for a page """
    return aggregate_max([f['form'] for f in forms_info(response)])


def available_form_types():
    """ Return a list of all possible form types """
    return list(classifiers.get_instance().form_classes)
