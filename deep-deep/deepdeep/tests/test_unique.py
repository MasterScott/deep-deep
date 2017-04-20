from scrapy.http.response.text import TextResponse  # type: ignore

from deepdeep.spiders.unique import UniqueContentGoal


def text_response(text, url='http://example.com'):
    return TextResponse(url, body=text.encode('utf8'), encoding='utf8')


def test_unique_goal():
    goal = UniqueContentGoal()
    assert goal.get_reward(text_response('some text')) == 0.03
    assert goal.get_reward(text_response('some text')) == 0.00
    assert goal.get_reward(text_response('some more text')) == 0.04
