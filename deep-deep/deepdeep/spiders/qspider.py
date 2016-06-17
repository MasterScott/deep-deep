# -*- coding: utf-8 -*-
import abc
import json
from pathlib import Path
from typing import Dict, Tuple, Union, Optional, List, Iterator

import joblib
import tqdm
import numpy as np
import scipy.sparse as sp
from formasaurus.utils import get_domain
import scrapy
from scrapy.http import TextResponse, Response
from scrapy.statscollectors import StatsCollector

from deepdeep.queues import (
    BalancedPriorityQueue,
    RequestsPriorityQueue,
    score_to_priority,
    priority_to_score)
from deepdeep.scheduler import Scheduler
from deepdeep.spiders._base import BaseSpider
from deepdeep.utils import (
    set_request_domain,
    url_path_query,
)
from deepdeep.qlearning import QLearner
from deepdeep.utils import log_time
from deepdeep.vectorizers import LinkVectorizer, PageVectorizer
from deepdeep.goals import FormasaurusGoal, BaseGoal


class QSpider(BaseSpider):
    """
    This spider learns how to crawl using Q-Learning.

    It starts from a list of seed URLs. When a page is received, spider

    1. updates Q function based on observed reward;
    2. extracts links and creates requests for them, using Q function
       to set priorities

    """
    name = 'q'
    _ARGS = {
        'double', 'use_urls', 'use_pages', 'use_same_domain',
        'eps', 'balancing_temperature', 'gamma',
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

    # whether to use URL path/query as a feature
    use_urls = 0

    # whether to use a 'link is to the same domain' feature
    use_same_domain = 1

    # whether to use page content as a feature
    use_pages = 0

    # use Double Learning
    double = 1

    # probability of selecting a random request
    eps = 0.2

    # 0 <= gamma < 1; lower values make spider focus on immediate reward.
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
    checkpoint_path = None  # type: Optional[str]

    # Is spider allowed to follow out-of-domain links?
    # XXX: it is not enough to set this to False; a middleware should be also
    # turned off.
    stay_in_domain = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.eps = float(self.eps)
        self.balancing_temperature = float(self.balancing_temperature)
        self.gamma = float(self.gamma)
        self.use_urls = bool(int(self.use_urls))
        self.use_pages = int(self.use_pages)
        self.use_same_domain = int(self.use_same_domain)
        self.double = int(self.double)
        self.stay_in_domain = bool(int(self.stay_in_domain))
        self.steps_before_switch = int(self.steps_before_switch)
        self.replay_sample_size = int(self.replay_sample_size)
        self.Q = QLearner(
            steps_before_switch=self.steps_before_switch,
            replay_sample_size=self.replay_sample_size,
            gamma=self.gamma,
            double_learning=bool(self.double),
            on_model_changed=self.on_model_changed,
        )
        self.link_vectorizer = LinkVectorizer(
            use_url=bool(self.use_urls),
            use_same_domain=bool(self.use_same_domain),
        )
        self.page_vectorizer = PageVectorizer()
        self.total_reward = 0
        self.model_changes = 0
        self.goal = self.get_goal()

        self.checkpoint_interval = int(self.checkpoint_interval)
        if self.checkpoint_path:
            params = json.dumps(self.get_params(), indent=4)
            print(params)
            (Path(self.checkpoint_path)/"params.json").write_text(params)

    # @abc.abstractmethod
    def get_goal(self) -> BaseGoal:
        """ This method should return a crawl goal object """
        # FIXME: remove hardcoded goal
        return FormasaurusGoal(formtype='password/login recovery')

    def is_seed(self, r: Union[scrapy.Request, Response]) -> bool:
        return 'link_vector' not in r.meta

    def parse(self, response):
        self.increase_response_count()
        self.close_finished_queues()
        self._debug_expected_vs_got(response)
        output = self._parse(response)
        if not self.is_seed(response):
            self.log_stats()
            self.maybe_checkpoint()
            yield self.get_stats_item()
        yield from output

    def _parse(self, response):
        if self.is_seed(response) and not hasattr(response, 'text'):
            # bad seed
            return []

        as_t = response.meta.get('link_vector')

        if not hasattr(response, 'text'):
            # learn to avoid non-html responses
            self.Q.add_experience(
                as_t=as_t,
                AS_t1=None,
                r_t1=0
            )
            return []

        page_vector = self._get_page_vector(response)
        links = self._extract_links(response)
        links_matrix = self.link_vectorizer.transform(links) if links else None
        links_matrix = self.Q.join_As(links_matrix, page_vector)

        if not self.is_seed(response):
            reward = self.goal.get_reward(response)
            self.total_reward += reward
            self.Q.add_experience(
                as_t=as_t,
                AS_t1=links_matrix,
                r_t1=reward
            )
            self.goal.response_observed(response)
        return list(self._links_to_requests(links, links_matrix))

    def _extract_links(self, response: TextResponse) -> List[Dict]:
        """ Return a list of all unique links on a page """
        return list(self.le.iter_link_dicts(
            response=response,
            limit_by_domain=self.stay_in_domain,
            deduplicate=False,
            deduplicate_local=True,
        ))

    def _links_to_requests(self,
                           links: List[Dict],
                           links_matrix: sp.csr_matrix,
                           ) -> Iterator[scrapy.Request]:
        indices_and_links = list(self.le.deduplicate_links_enumerated(links))
        if not indices_and_links:
            return
        indices, links_to_follow = zip(*indices_and_links)
        AS = links_matrix[list(indices)]
        scores = self.Q.predict(AS)

        for link, v, score in zip(links_to_follow, AS, scores):
            url = link['url']
            next_domain = get_domain(url)
            meta = {
                'link_vector': v,
                'link': link,  # FIXME: turn it off for production
                'scheduler_slot': next_domain,
            }
            priority = score_to_priority(score)
            req = scrapy.Request(url, priority=priority, meta=meta)
            set_request_domain(req, next_domain)
            yield req

    def _get_page_vector(self, response: TextResponse) -> Optional[np.ndarray]:
        """ Convert response content to a feature vector """
        if not self.use_pages:
            return None
        return self.page_vectorizer.transform([response.text])[0]

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

    def on_model_changed(self):
        self.model_changes += 1
        if (self.model_changes % 1) == 0:
            self.recalculate_request_priorities()

    def close_finished_queues(self):
        for slot in self.scheduler.queue.get_active_slots():
            if self.goal.is_acheived_for(domain=slot):
                self.scheduler.close_slot(slot)

    @log_time
    def recalculate_request_priorities(self):
        # TODO: vectorize
        def request_priority(request: scrapy.Request) -> int:
            if self.is_seed(request):
                return request.priority

            as_ = request.meta['link_vector']
            score = self.Q.predict_one(as_)
            if score > 0.5 and 'link' in request.meta:
                self._log_promising_link(request.meta['link'], score)
            return score_to_priority(score)

        for slot in tqdm.tqdm(self.scheduler.queue.get_active_slots()):
            queue = self.scheduler.queue.get_queue(slot)
            queue.update_all_priorities(request_priority)

    def _log_promising_link(self, link, score):
        self.logger.debug("PROMISING LINK {:0.4f}: {}\n        {}".format(
            score, link['url'], link['inside_text']
        ))

    def _examples(self):
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
            ['forums', 'http://other-domain.com/mybb'],
            ['sadhjgrhgsfd', 'http://example.com/new-to-exhibiting/discover-your-stand-position/'],
            ['забыли пароль', 'http://example.com/users/send-password/'],
        ]
        examples_repr = [
            "{:20s} {}".format(txt, url)
            for txt, url in examples
        ]
        links = [
            {
                'inside_text': txt,
                'url': url,
                'domain_from': 'example',
                'domain_to': get_domain(url),
            }
            for txt, url in examples
        ]
        A = self.link_vectorizer.transform(links)
        s = self.page_vectorizer.transform([""]) if self.use_pages else None
        AS = self.Q.join_As(A, s)
        return examples_repr, AS

    def log_stats(self):
        examples, AS = self._examples()
        if examples:
            scores_target = self.Q.predict(AS)
            scores_online = self.Q.predict(AS, online=True)
            for ex, score1, score2 in zip(examples, scores_target, scores_online):
                print(" {:0.4f} {:0.4f} {}".format(score1, score2, ex))

        print("t={}, return={:0.4f}, avg return={:0.4f}, L2 norm: {:0.4f} {:0.4f}".format(
            self.Q.t_,
            self.total_reward,
            self.total_reward / self.Q.t_ if self.Q.t_ else 0,
            self.Q.coef_norm(online=True),
            self.Q.coef_norm()
        ))
        self.goal.debug_print()

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
            '_type': 'stats',
            't': self.Q.t_,
            'return': self.total_reward,
            'domains_open': domains_open,
            'domains_closed': domains_closed,
            'enqueued': enqueued,
            'processed': dequeued,
            'dropped': dropped,
            'todo': todo,
        }

    def _debug_expected_vs_got(self, response: Response):
        if 'link' not in response.meta:
            return
        reward = self.goal.get_reward(response)
        self.logger.debug("\nGOT {:0.4f} (expected return was {:0.4f}) {}\n{}".format(
            reward,
            priority_to_score(response.request.priority),
            response.url,
            response.meta['link'].get('inside_text'),
        ))

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
            'link_vectorizer': self.link_vectorizer,
            'page_vectorizer': self.page_vectorizer,
            '_params': self.get_params(),
        }
        joblib.dump(data, str(path), compress=3)
