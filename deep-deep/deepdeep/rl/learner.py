# -*- coding: utf-8 -*-
from __future__ import absolute_import
from typing import Callable, Optional

import joblib
import numpy as np
from scipy import sparse
import sklearn.base
from sklearn.linear_model import SGDRegressor

from deepdeep.rl.experience import ExperienceMemory
from deepdeep.utils import log_time


class QLearner:
    """
    Q-learning estimator with function approximation, experience replay
    and double learning.

    Todo: Q(s, a) instead of just Q(a).
    """
    def __init__(self,
                 double_learning: bool = True,
                 steps_before_switch: int = 100,
                 gamma: float = 0.3,
                 initial_predictions: float = 0.05,
                 sample_size: int = 300,
                 on_model_changed: Optional[Callable[[], None]]=None,
                 ) -> None:
        assert 0 <= gamma < 1
        self.double_learning = double_learning
        self.steps_before_switch = steps_before_switch
        self.gamma = gamma
        self.initial_predictions = initial_predictions
        self.sample_size = sample_size
        self.on_model_changed = on_model_changed

        self.clf_online = SGDRegressor(
            penalty='l2',
            average=False,
            n_iter=1,
            learning_rate='constant',
            # loss='epsilon_insensitive',
            alpha=1e-6,
            eta0=0.1,
        )

        self.clf_target = sklearn.base.clone(self.clf_online)  # type: SGDRegressor
        self.memory = ExperienceMemory()
        self.t_ = 0

    def add_experience(self, a_t, A_t1, r_t1) -> None:
        self.t_ += 1
        self.memory.add(
            a_t=a_t,
            A_t1=A_t1,
            r_t1=r_t1,
        )
        self.fit_iteration(self.sample_size)
        if (self.t_ % self.steps_before_switch) == 0:
            self._update_target_clf()
            if self.on_model_changed is not None:
                self.on_model_changed()

    def predict(self, A, online=False) -> np.ndarray:
        clf = self.clf_target if not online else self.clf_online
        if clf.coef_ is None:
            return np.ones(A.shape[0]) * self.initial_predictions
        return clf.predict(A)

    def predict_one(self, a, online=False) -> float:
        return self.predict(sparse.vstack([a]), online=online)[0]

    @log_time
    def fit_iteration(self, sample_size: int) -> None:
        sample = self.memory.sample(sample_size)
        a_t_list, A_t1_list, r_t1_list = zip(*sample)
        rewards = np.asarray(r_t1_list)

        Q_t1_values = np.zeros_like(rewards)
        for idx, A_t1 in enumerate(A_t1_list or []):
            if A_t1 is not None and A_t1.shape[0] > 0:
                scores = self.predict(A_t1, online=True)
                if self.double_learning:
                    # This is a simple variant of double learning
                    # used in http://arxiv.org/abs/1509.06461.
                    # Instead of using totally separate Q functions
                    # action is chosen by online Q function, but the score
                    # is estimated using target Q function.
                    best_idx = scores.argmax()
                    a_t1 = A_t1[best_idx]
                    Q_t1_values[idx] = self.predict_one(a_t1, online=False)
                else:
                    Q_t1_values[idx] = scores.max()  # vanilla Q-learning

        X = sparse.vstack(a_t_list)
        y = rewards + self.gamma * Q_t1_values
        self.clf_online.partial_fit(X, y)

    def _update_target_clf(self):
        trained_params = [
            't_',
            'coef_',
            'intercept_',
            'average_coef_',
            'average_intercept_',
            'standard_coef_',
            'standard_intercept_',
        ]
        for attr in trained_params:
            if not hasattr(self.clf_online, attr):
                continue
            data = getattr(self.clf_online, attr)
            if hasattr(data, 'copy'):
                data = data.copy()
            setattr(self.clf_target, attr, data)

    def coef_norm(self, online: bool=True) -> float:
        clf = self.clf_target if not online else self.clf_online
        if clf.coef_ is None:
            return 0
        return np.sqrt((clf.coef_ ** 2).sum())

    def __getstate__(self):
        dct = self.__dict__.copy()
        del dct['on_model_changed']
        return dct
