#!/usr/bin/env python
"""
Train LDA model on CDR data.

Usage:
    show-lda-topics.py <model.joblib> [--top=<N>]
    show-lda-topics.py visualize <model.joblib> <cdritems.jl.gz> <out.html>

Options:
    --top <N>   Print top N words for each topic [default: 20]

"""
import sys
from pathlib import Path
sys.path.insert(0, str((Path(__file__).parent / "..").absolute()))

import joblib
from docopt import docopt
from tqdm import tqdm
from sklearn.pipeline import Pipeline
import json_lines


def iter_html(path):
    with json_lines.open(path, broken=True) as lines:
        for line in lines:
            yield line['raw_content']


def print_top_words(model, feature_names, n_top_words, word_threshold=0.11):
    def _weights_repr(topic, indices):
        return ["{}".format(feature_names[i]) for i in indices
                if topic[i] > word_threshold]

    topics = sorted(enumerate(model.components_),
                    key=lambda idx_c: idx_c[1].sum(), reverse=True)

    for idx, (topic_idx, topic) in enumerate(topics):
        print("%d) Topic #%d:" % (idx, topic_idx))
        neg_tokens = topic.argsort()
        pos_tokens = neg_tokens[::-1][:n_top_words]
        print("Weight: %0.1f total, %0.1f top" % (
            topic.sum(), topic[pos_tokens].sum()))
        print(" ".join(_weights_repr(topic, pos_tokens)))
        print(" ")
    print()


def main(args):
    pipe = joblib.load(args['<model.joblib>'])  # type: Pipeline
    if len(pipe.steps) != 3:
        for name, step in pipe.steps:
            print(name)
            print(step)
        raise ValueError("Unsupported pipeline: %s" % pipe)

    vec, lda, norm = [s[1] for s in pipe.steps]

    if args['visualize']:
        import pyLDAvis.sklearn
        X = vec.transform(tqdm(iter_html(args['<cdritems.jl.gz>'])))
        p = pyLDAvis.sklearn.prepare(lda, X, vec)
        pyLDAvis.save_html(p, args['<out.html>'])
    else:
        print_top_words(lda, vec.get_feature_names(), int(args['--top']))


if __name__ == '__main__':
    main(docopt(__doc__))
