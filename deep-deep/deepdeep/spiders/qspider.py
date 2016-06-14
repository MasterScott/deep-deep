# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import Dict, Tuple
from weakref import WeakKeyDictionary

import joblib
import tqdm
import scrapy
from scrapy.http import TextResponse
from scrapy.statscollectors import StatsCollector

from deepdeep.queues import (
    BalancedPriorityQueue,
    RequestsPriorityQueue,
    score_to_priority,
    priority_to_score)
from deepdeep.scheduler import Scheduler
from deepdeep.spiders._base import BaseSpider
from deepdeep.utils import (
    get_response_domain,
    set_request_domain,
    url_path_query,
    MaxScores,
)
from deepdeep.score_pages import response_max_scores
from deepdeep.qlearning import QLearner
from deepdeep.utils import log_time
from deepdeep.vectorizers import LinkVectorizer


class FormasaurusGoal:
    """
    Parameters
    ----------

    formtype : str
        Form type to look for. Allowed values:

        * "search"
        * "login"
        * "registration"
        * "password/login recovery"
        * "contact/comment"
        * "join mailing list"
        * "order/add to cart"
        * "other"

    threshold : float
         Probability threshold required to consider the goal acheived
         for a domain (default: 0.7).
    """
    def __init__(self, formtype: str, threshold: float=0.7) -> None:
        self.formtype = formtype
        self.threshold = threshold
        self._cache = WeakKeyDictionary()
        self._domain_scores = MaxScores()  # domain -> max score

    def get_reward(self, response: TextResponse) -> float:
        """ Return a reward for a response """
        if response not in self._cache:
            if hasattr(response, 'text'):
                scores = response_max_scores(response)
                score = scores.get(self.formtype, 0.0)
                # score = score if score > 0.5 else 0
            else:
                score = 0.0
            self._cache[response] = score
        return self._cache[response]

    def response_observed(self, response: TextResponse):
        reward = self.get_reward(response)
        domain = get_response_domain(response)
        self._domain_scores.update(domain, reward)

    def is_acheived_for(self, domain):
        score = self._domain_scores[domain]
        return score > self.threshold

    def domain_score(self, domain):
        return self._domain_scores[domain]

    def print_score_stats(self):
        print("Scores: sum={:8.1f}, avg={:0.4f}".format(
            self._domain_scores.sum(), self._domain_scores.avg()
        ))


class QSpider(BaseSpider):
    name = 'q'

    _ARGS = {
        'double', 'task', 'use_urls', 'eps', 'balancing_temperature', 'gamma',
        'replay_sample_size', 'steps_before_switch',
        'checkpoint_path', 'checkpoint_interval',
    }
    ALLOWED_ARGUMENTS = _ARGS | BaseSpider.ALLOWED_ARGUMENTS
    custom_settings = {
        'DEPTH_LIMIT': 10,
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

    # 0 <= gamma < 1; lower values make spider focus on immediate reward.
    #
    #     gamma     % of credit assigned to n-th previous step  effective steps
    #     -----     ------------------------------------------  ---------------
    #     0.00      100   0   0   0   0   0   0   0   0   0     1
    #     0.05      100   5   0   0   0   0   0   0   0   0     2
    #     0.10      100  10   1   0   0   0   0   0   0   0     2
    #     0.15      100  15   2   0   0   0   0   0   0   0     2
    #     0.20      100  20   4   0   0   0   0   0   0   0     2
    #     0.25      100  25   6   1   0   0   0   0   0   0     3
    #     0.30      100  30   9   2   0   0   0   0   0   0     3
    #     0.35      100  35  12   4   1   0   0   0   0   0     3
    #     0.40      100  40  16   6   2   1   0   0   0   0     4
    #     0.45      100  45  20   9   4   1   0   0   0   0     4
    #     0.50      100  50  25  12   6   3   1   0   0   0     5
    #     0.55      100  55  30  16   9   5   2   1   0   0     6
    #     0.60      100  60  36  21  12   7   4   2   1   1     6
    #     0.65      100  65  42  27  17  11   7   4   3   2     7
    #     0.70      100  70  48  34  24  16  11   8   5   4     9
    #     0.75      100  75  56  42  31  23  17  13  10   7     10
    #     0.80      100  80  64  51  40  32  26  20  16  13     10+
    #     0.85      100  85  72  61  52  44  37  32  27  23     10+
    #     0.90      100  90  81  72  65  59  53  47  43  38     10+
    #     0.95      100  95  90  85  81  77  73  69  66  63     10+
    #
    gamma = 0.4

    # softmax temperature for domain balancer;
    # higher values => more randomeness in domain selection.
    balancing_temperature = 1.0

    # parameters of online Q function are copied to target Q function
    # every `steps_before_switch` steps
    steps_before_switch = 100

    # how many examples to fetch from experience replay on each iteration
    replay_sample_size = 300

    # current model is saved every checkpoint_interval timesteps
    checkpoint_interval = 1000

    # Where to store checkpoints. By default they are not stored.
    checkpoint_path = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.eps = float(self.eps)
        self.balancing_temperature = float(self.balancing_temperature)
        self.gamma = float(self.gamma)
        self.use_urls = bool(int(self.use_urls))
        self.double = bool(int(self.double))
        self.steps_before_switch = int(self.steps_before_switch)
        self.replay_sample_size = int(self.replay_sample_size)
        self.checkpoint_interval = int(self.checkpoint_interval)

        self.goal = FormasaurusGoal(formtype=self.task)
        self.Q = QLearner(
            steps_before_switch=self.steps_before_switch,
            replay_sample_size=self.replay_sample_size,
            gamma=self.gamma,
            double_learning=self.double,
            on_model_changed=self.on_model_changed,
        )
        self.link_vec = LinkVectorizer(use_url=self.use_urls)
        self.total_reward = 0
        self.model_changes = 0

        params = json.dumps(self.get_params(), indent=4)
        print(params)
        (Path(self.checkpoint_path)/"params.json").write_text(params)

    def on_model_changed(self):
        self.model_changes += 1
        if (self.model_changes % 1) == 0:
            self.recalculate_request_priorities()

    def close_finished_queues(self):
        for slot in self.scheduler.queue.get_active_slots():
            if self.goal.is_acheived_for(domain=slot):
                score = self.goal.domain_score(slot)
                print("Queue {} is closed; score={:0.4f}.".format(slot, score))
                self.scheduler.close_slot(slot)

    def parse(self, response):
        self.increase_response_count()
        self.close_finished_queues()

        if 'link' in response.meta:
            reward = self.goal.get_reward(response)
            self.logger.debug("\nGOT {:0.4f} (expected return was {:0.4f}) {}\n{}".format(
                reward,
                priority_to_score(response.request.priority),
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
                self.maybe_checkpoint()
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
            reward = self.goal.get_reward(response)
            self.total_reward += reward
            self.Q.add_experience(
                a_t=response.meta['link_vector'],
                A_t1=links_matrix,
                r_t1=reward
            )
            self.log_stats()
            self.maybe_checkpoint()
            yield self.get_stats_item()
            self.goal.response_observed(response)

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
                    }
                    priority = score_to_priority(score)
                    req = scrapy.Request(link['url'], priority=priority, meta=meta)
                    set_request_domain(req, domain)
                    if score > 0.5:
                        self._log_promising_link(link, score)
                    yield req

    def get_scheduler_queue(self):
        """
        This method is called by deepdeep.scheduler.Scheduler
        to create a new queue.
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
        self.logger.debug("PROMISING LINK {:0.4f}: {}\n        {}".format(
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
        self.goal.print_score_stats()

        stats = self.get_stats_item()
        print("Domains: {domains_open} open, {domains_closed} closed; "
              "{todo} requests in queue, {processed} processed, {dropped} dropped".format(**stats))

    def get_stats_item(self):
        domains_open, domains_closed = self._domain_stats()
        stats = self.crawler.stats  # type: StatsCollector
        enqueued = stats.get_value('custom-scheduler/enqueued/', 0)
        dequeued = stats.get_value('custom-scheduler/dequeued/', 0)
        dropped = stats.get_value('custom-scheduler/dropped/', 0)
        todo = enqueued - dequeued - dropped

        return {
            't': self.Q.t_,
            'return': self.total_reward,
            'domains_open': domains_open,
            'domains_closed': domains_closed,
            'enqueued': enqueued,
            'processed': dequeued,
            'dropped': dropped,
            'todo': todo,
        }

    def _domain_stats(self) -> Tuple[int, int]:
        domains_open = len(self.scheduler.queue.get_active_slots())
        domains_closed = len(self.scheduler.queue.closed_slots)
        return domains_open, domains_closed

    def get_params(self):
        keys = self._ARGS - {'checkpoint_path', 'checkpoint_interval'}
        return {key: getattr(self, key) for key in keys}

    def maybe_checkpoint(self):
        if not self.checkpoint_path:
            return
        if (self.Q.t_ % self.checkpoint_interval) != 0:
            return
        path = Path(self.checkpoint_path)
        filename = "Q-%s.joblib" % self.Q.t_
        self.dump_policy(path.joinpath(filename))

    @log_time
    def dump_policy(self, path):
        """ Save the current policy """
        data = {
            'Q': self.Q,
            'link_vec': self.link_vec,
            '_params': self.get_params(),
        }
        joblib.dump(data, str(path), compress=3)
