import re
from typing import Dict, Callable

import tensorflow as tf


_session = None  # type: tf.Session
_writer = None  # type: tf.train.SummaryWriter
_loggers = {}  # type: Dict[str, Callable[float, int]]


def configure(logdir):
    global _session, _writer
    _session = tf.Session()
    _writer = tf.train.SummaryWriter(logdir, flush_secs=2)


def log_value(name: str, value: float, step: int):
    if _session is None:
        return
    name = '_'.join(re.findall('\w+', name))
    logger = _loggers.get(name)
    if logger is None:

        dtype = tf.float32
        variable = tf.Variable(
            initial_value=value, dtype=dtype, trainable=False, name=name)
        _session.run(tf.initialize_variables([variable], name))
        summary_op = tf.scalar_summary(name, variable)
        new_value = tf.placeholder(dtype, shape=[])
        assign_op = tf.assign(variable, new_value)

        def logger(x, i):
            _, summary = _session.run([assign_op, summary_op], {new_value: x})
            _writer.add_summary(summary, i)

        _loggers[name] = logger

    logger(value, step)
