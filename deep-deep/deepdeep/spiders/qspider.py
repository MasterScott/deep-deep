# -*- coding: utf-8 -*-
import time

import numpy as np
from sklearn.linear_model import SGDRegressor
import scrapy
from scipy import sparse
import tqdm

from deepdeep.queues import (
    BalancedPriorityQueue,
    RequestsPriorityQueue,
    FLOAT_PRIORITY_MULTIPLIER
)
from deepdeep.spiders.base import BaseSpider
from deepdeep.utils import (
    get_response_domain,
    set_request_domain,
    MaxScores,
)
from deepdeep.score_pages import (
    response_max_scores,
)
from deepdeep.rl.experience import ExperienceMemory
from sklearn.feature_extraction.text import HashingVectorizer
import sklearn.base

from formasaurus.text import normalize


def _link_inside_text(link):
    text = link.get('inside_text', '')
    title = link.get('attrs', {}).get('title', '')
    return normalize(text + ' ' + title)


def LinkVectorizer():
    return HashingVectorizer(
        preprocessor=_link_inside_text,
        ngram_range=(1,2),
        n_features=100*1024,
        binary=True,
        norm='l2',
    )


def log_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            end = time.time()
            print("{} took {:0.4f}s".format(func, end-start))
    return wrapper


class QLearner:
    """
    Q-learning estimator with function approximation, experience replay
    and double learning.

    Todo: Q(s, a) instead of just Q(a).
    """
    def __init__(self,
                 double=True,
                 steps_before_switch=100,
                 gamma=0.3,
                 initial_predictions=0.05,
                 sample_size=300,
                 on_model_changed=None,
                 ):
        assert double is True, "double=False is not implemented"
        assert 0 <= gamma <= 1
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

    def add_experience(self, a_t, A_t1, r_t1):
        self.t_ += 1
        self.memory.add(
            a_t=a_t,
            A_t1=A_t1,
            r_t1=r_t1,
        )
        self.fit_iteration(self.sample_size)
        if (self.t_ % self.steps_before_switch) == 0:
            self._update_target_clf()
            if self.on_model_changed:
                self.on_model_changed()

    def predict(self, A, online=False):
        clf = self.clf_target if not online else self.clf_online
        if clf.coef_ is None:
            return np.ones(A.shape[0]) * self.initial_predictions
        return clf.predict(A)

    def predict_one(self, a, online=False):
        return self.predict(sparse.vstack([a]), online=online)[0]

    @log_time
    def fit_iteration(self, sample_size):
        sample = self.memory.sample(sample_size)
        a_t_list, A_t1_list, r_t1_list = zip(*sample)
        rewards = np.asarray(r_t1_list)

        Q_t1_values = np.zeros_like(rewards)
        for idx, A_t1 in enumerate(A_t1_list or []):
            # TODO: more vectorization
            if A_t1 is not None:
                scores = self.predict(A_t1, online=True)
                best_idx = scores.argmax()
                a_t1 = A_t1[best_idx]
                Q_t1_values[idx] = self.predict_one(a_t1, online=False)

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

    def coef_norm(self, online=True):
        clf = self.clf_target if not online else self.clf_online
        if clf.coef_ is None:
            return 0
        return np.sqrt((clf.coef_ ** 2).sum())


def score_to_priority(score: float) -> int:
    return int(score * FLOAT_PRIORITY_MULTIPLIER)


class QSpider(BaseSpider):
    name = 'q'

    custom_settings = {
        'DEPTH_LIMIT': 5,
        # 'SPIDER_MIDDLEWARES': {
        #     'deepdeep.spidermiddlewares.CrawlGraphMiddleware': 400,
        # }
    }
    initial_priority = score_to_priority(5)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Q = QLearner(
            # steps_before_switch=100,
            on_model_changed=self.on_model_changed,
        )
        self.link_vec = LinkVectorizer()
        self.total_reward = 0
        self.model_changes = 0
        self.domain_scores = MaxScores(['score'])

    def on_model_changed(self):
        self.model_changes += 1
        if (self.model_changes % 1) == 0:
            self.recalculate_request_priorities()

    def get_scheduler_queue(self):
        """
        This method is called by scheduler to create a new queue.
        """
        def new_queue(domain):
            return RequestsPriorityQueue(fifo=True)
        return BalancedPriorityQueue(queue_factory=new_queue, eps=0.2)

    @property
    def scheduler_queue(self) -> BalancedPriorityQueue:
        return self.crawler.engine.slot.scheduler.queue

    @log_time
    def recalculate_request_priorities(self):
        # TODO: vectorize
        def request_priority(request: scrapy.Request):
            link_vector = request.meta.get('link_vector', None)
            if link_vector is None:
                return request.priority
            score = self.Q.predict_one(link_vector)
            return score_to_priority(score)

        for slot in tqdm.tqdm(self.scheduler_queue.get_active_slots()):
            queue = self.scheduler_queue.get_queue(slot)
            queue.update_all_priorities(request_priority)

    def get_reward(self, response):
        scores = response_max_scores(response)
        # scores.get('registration', 0.0) +
        return scores.get('password/login recovery', 0.0)
        # return scores.get('login', 0.0)

    def parse(self, response):
        self.increase_response_count()

        domain = get_response_domain(response)
        links = list(self.iter_link_dicts(
            response=response,
            domain=domain,
            deduplicate=False
        ))
        links_matrix = self.link_vec.transform(links) if links else None

        if 'link_vector' in response.meta:
            reward = self.get_reward(response)
            self.total_reward += reward
            self.Q.add_experience(
                a_t=response.meta['link_vector'],
                A_t1=links_matrix,
                r_t1=reward
            )
            self.debug_Q()
            self.domain_scores.update(domain, {'score': reward})

        if links:
            _links = list(self.deduplicate_links(links, indices=True))
            if _links:
                indices, links_to_follow = zip(*_links)
                links_to_follow_matrix = links_matrix[list(indices)]
                scores = self.Q.predict(links_to_follow_matrix)

                for link, v, score in zip(links_to_follow, links_to_follow_matrix, scores):
                    meta = {
                        'link_vector': v,
                        'scheduler_slot': domain,
                    }
                    priority = score_to_priority(score)
                    req = scrapy.Request(link['url'], priority=priority, meta=meta)
                    set_request_domain(req, domain)
                    yield req

    def debug_Q(self):
        examples = [
            'forgot password',
            'registration',
            'register',
            'sign up',
            'my account',
            'my little pony',
            'comment',
            'sign in',
            'login',
            'forum',
            'sadhjgrhgsfd',
            'забыли пароль'
        ]
        links = [{'inside_text': e} for e in examples]
        A = self.link_vec.transform(links)
        scores_target = self.Q.predict(A)
        scores_online = self.Q.predict(A, online=True)
        for ex, score1, score2 in zip(examples, scores_target, scores_online):
            print("{:20s} {:0.4f} {:0.4f}".format(ex, score1, score2))

        print("t={}, return={:0.4f}, avg return={:0.4f}, L2 norm: {:0.4f} {:0.4f}".format(
            self.Q.t_,
            self.total_reward,
            self.total_reward / self.Q.t_ if self.Q.t_ else 0,
            self.Q.coef_norm(online=True),
            self.Q.coef_norm()
        ))

        scores_sum = sorted(self.domain_scores.sum().items())
        scores_avg = sorted(self.domain_scores.avg().items())
        reward_lines = [
            "{:8.1f}   {:0.4f}   {}".format(tot, avg, k)
            for ((k, tot), (k, avg)) in zip(scores_sum, scores_avg)
        ]
        msg = '\n'.join(reward_lines)
        print(msg)
