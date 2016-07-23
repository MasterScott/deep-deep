#!/usr/bin/env python
"""
Train LDA model on CDR data.

Usage:
    train-lda.py <cdr_items.jl.gz> <output.joblib> [--n-topics=<N>] [--max-features=<N>]

Options:
    --n-topics=<N>      A number of LDA topics [default: 50]
    --max-features=<N>  Maximum number of features [default: 100000]

"""
import sys
from pathlib import Path
sys.path.insert(0, str((Path(__file__).parent / "..").absolute()))

from docopt import docopt
import joblib
from tqdm import tqdm

from deepdeep.vectorizers import LDAPageVctorizer
from deepdeep.utils import iter_jsonlines


def iter_html(path):
    return (line['raw_content'] for line in iter_jsonlines(path))


def train(cdr_jlgz, n_topics=50, batch_size=1024, min_df=4, max_features=None):
    lda_pipe = LDAPageVctorizer(
        n_topics=n_topics,
        batch_size=batch_size,
        min_df=min_df,
        verbose=1,
        max_features=max_features or None,
    )
    lda_pipe.fit(tqdm(iter_html(cdr_jlgz), desc="Loading HTML"))
    for name, step in lda_pipe.steps:
        step.verbose = False
    return lda_pipe


def main(args):
    max_features = args['--max-features']
    if max_features is not None:
        max_features = int(max_features)
    pipe = train(
        args['<cdr_items.jl.gz>'],
        n_topics=int(args['--n-topics']),
        max_features=max_features,
    )
    joblib.dump(pipe, args['<output.joblib>'], compress=3)


if __name__ == '__main__':
    main(docopt(__doc__))
