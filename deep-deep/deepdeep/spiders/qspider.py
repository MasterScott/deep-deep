# -*- coding: utf-8 -*-
from typing import Dict, Tuple

import tqdm
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import make_union
from formasaurus.text import normalize
import scrapy
from scrapy.http import TextResponse
from scrapy.utils.url import canonicalize_url

from deepdeep.queues import (
    BalancedPriorityQueue,
    RequestsPriorityQueue,
    FLOAT_PRIORITY_MULTIPLIER
)
from deepdeep.scheduler import Scheduler
from deepdeep.spiders.base import BaseSpider
from deepdeep.utils import (
    get_response_domain,
    set_request_domain,
    url_path_query,
    MaxScores,
)
from deepdeep.score_pages import response_max_scores
from deepdeep.rl.learner import QLearner
from deepdeep.utils import log_time


def _link_inside_text(link: Dict):
    text = link.get('inside_text', '')
    title = link.get('attrs', {}).get('title', '')
    return normalize(text + ' ' + title)


def _clean_url(link: Dict):
    return url_path_query(canonicalize_url(link.get('url')))


def LinkVectorizer(use_url: bool=False):
    text_vec = HashingVectorizer(
        preprocessor=_link_inside_text,
        n_features=1024*1024,
        binary=True,
        norm='l2',
        # ngram_range=(1, 2),
        analyzer='char',
        ngram_range=(3, 5),
    )
    if not use_url:
        return text_vec

    url_vec = HashingVectorizer(
        preprocessor=_clean_url,
        n_features=1024*1024,
        binary=True,
        analyzer='char',
        ngram_range=(4,5),
    )
    return make_union(text_vec, url_vec)


def score_to_priority(score: float) -> int:
    return int(score * FLOAT_PRIORITY_MULTIPLIER)


class QSpider(BaseSpider):
    name = 'q'
    ALLOWED_ARGUMENTS = (
        {'double', 'task', 'use_urls', 'eps', 'balancing_temperature', 'gamma'} |
        BaseSpider.ALLOWED_ARGUMENTS
    )
    custom_settings = {
        'DEPTH_LIMIT': 5,
        # 'SPIDER_MIDDLEWARES': {
        #     'deepdeep.spidermiddlewares.CrawlGraphMiddleware': 400,
        # }
    }
    initial_priority = score_to_priority(5)

    # Goal. Allowed values:
    #     "search"
    #     "login"
    #     "registration"
    #     "password/login recovery"
    #     "contact/comment"
    #     "join mailing list"
    #     "order/add to cart"
    #     "other"
    task = 'password/login recovery'

    # whether to use URL path/query as a feature
    use_urls = 0

    # use Double Learning
    double = 1

    # probability of selecting a random request
    eps = 0.2

    # 0 <= gamma <= 1; lower values make spider focus on immediate reward.
    gamma = 0.3

    # softmax temperature for domain balancer;
    # higher values => more randomeness.
    balancing_temperature = 1.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.eps = float(self.eps)
        self.balancing_temperature = float(self.balancing_temperature)
        self.gamma = float(self.gamma)

        self.Q = QLearner(
            gamma=self.gamma,
            double_learning=bool(int(self.double)),
            on_model_changed=self.on_model_changed,
        )
        self.link_vec = LinkVectorizer(bool(int(self.use_urls)))
        self.total_reward = 0
        self.model_changes = 0
        self.domain_scores = MaxScores(['score'])

    def on_model_changed(self):
        self.model_changes += 1
        if (self.model_changes % 1) == 0:
            self.recalculate_request_priorities()

    def get_reward(self, response: TextResponse) -> float:
        if not hasattr(response, 'text'):
            return 0.0
        scores = response_max_scores(response)
        return scores.get(self.task, 0.0)

    def close_finished_queues(self):
        for slot in self.scheduler.queue.get_active_slots():
            score = self.domain_scores[slot]['score']
            if score > 0.7:
                print("Queue {} is closed; score={:0.4f}.".format(slot, score))
                self.scheduler.close_slot(slot)

    def parse(self, response):
        self.increase_response_count()
        self.close_finished_queues()

        if 'link' in response.meta:
            reward = self.get_reward(response)
            self.logger.info("\nGOT {:0.4f} (expected return was {:0.4f}) {}\n{}".format(
                reward,
                response.request.priority / FLOAT_PRIORITY_MULTIPLIER,
                response.url,
                response.meta['link'].get('inside_text'),
            ))

        if not hasattr(response, 'text'):
            if 'link_vector' in response.meta:
                # learn to avoid non-html responses
                self.Q.add_experience(
                    a_t=response.meta['link_vector'],
                    A_t1=None,
                    r_t1=0
                )
                self.log_stats()
                yield self.get_stats_item()
            return

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
            self.log_stats()
            yield self.get_stats_item()
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
                        'link': link,  # FIXME: turn it off for production
                        'scheduler_slot': domain,
                        'domain': domain,  # stay on-domain
                    }
                    priority = score_to_priority(score)
                    req = scrapy.Request(link['url'], priority=priority, meta=meta)
                    set_request_domain(req, domain)
                    if score > 0.5:
                        self._log_promising_link(link, score)
                    yield req

    def get_scheduler_queue(self):
        """
        This method is called by scheduler to create a new queue.
        """
        def new_queue(domain):
            return RequestsPriorityQueue(fifo=True)
        return BalancedPriorityQueue(
            queue_factory=new_queue,
            eps=self.eps,
            balancing_temperature=self.balancing_temperature,
        )

    @property
    def scheduler(self) -> Scheduler:
        return self.crawler.engine.slot.scheduler

    @log_time
    def recalculate_request_priorities(self):
        # TODO: vectorize
        def request_priority(request: scrapy.Request):
            link_vector = request.meta.get('link_vector', None)
            if link_vector is None:
                return request.priority
            score = self.Q.predict_one(link_vector)
            if score > 0.5:
                self._log_promising_link(request.meta['link'], score)
            return score_to_priority(score)

        for slot in tqdm.tqdm(self.scheduler.queue.get_active_slots()):
            queue = self.scheduler.queue.get_queue(slot)
            queue.update_all_priorities(request_priority)

    def _log_promising_link(self, link, score):
        self.logger.info("PROMISING LINK {:0.4f}: {}\n        {}".format(
            score, link['url'], link['inside_text']
        ))

    def log_stats(self):
        examples = [
            ['forgot password', 'http://example.com/wp-login.php?action=lostpassword'],
            ['registration', 'http://example.com/register'],
            ['register', 'http://example.com/reg'],
            ['sign up', 'http://example.com/users/new'],
            ['my account', 'http://example.com/account/my?sess=GJHFHJS21123'],
            ['my little pony', 'http://example.com?category=25?sort=1&'],
            ['comment', 'http://example.com/blog?p=2'],
            ['sign in', 'http://example.com/users/login'],
            ['login', 'http://example.com/users/login'],
            ['forum', 'http://example.com/mybb'],
            ['forums', 'http://example.com/mybb'],
            ['sadhjgrhgsfd', 'http://example.com/new-to-exhibiting/discover-your-stand-position/'],
            ['забыли пароль', 'http://example.com/users/send-password/'],
        ]
        links = [{'inside_text': txt, 'url': url} for txt, url in examples]
        A = self.link_vec.transform(links)
        scores_target = self.Q.predict(A)
        scores_online = self.Q.predict(A, online=True)
        for (txt, url), score1, score2 in zip(examples, scores_target, scores_online):
            print(" {:0.4f} {:0.4f} {:20s} {}".format(
                score1, score2, txt, url_path_query(url),
            ))

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

        domains_open, domains_closed = self._domain_stats()
        print("Domains: {} open, {} closed".format(domains_open, domains_closed))

    def get_stats_item(self):
        domains_open, domains_closed = self._domain_stats()
        return {
            't': self.Q.t_,
            'return': self.total_reward,
            'domains_open': domains_open,
            'domains_closed': domains_closed,
        }

    def _domain_stats(self) -> Tuple[int, int]:
        domains_open = len(self.scheduler.queue.get_active_slots())
        domains_closed = len(self.scheduler.queue.closed_slots)
        return domains_open, domains_closed
