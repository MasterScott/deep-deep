# -*- coding: utf-8 -*-
from __future__ import absolute_import

from .qspider import QSpider
from deepdeep.goals import FormasaurusGoal
from deepdeep.utils import get_domain


class FormSpider(QSpider):
    """
    This spider learns how to crawl relevant pages.
    """
    name = 'forms'

    def get_goal(self) -> FormasaurusGoal:
        return FormasaurusGoal(formtype='password/login recovery')

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
