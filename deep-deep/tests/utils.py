import pytest
from scrapy.utils.log import configure_logging
from scrapy.utils.python import to_bytes
from twisted.internet import defer
from twisted.web.resource import Resource


# make the module importable without running py.test (for mockserver)
try:
    inlineCallbacks = pytest.inlineCallbacks
except AttributeError:
    inlineCallbacks = defer.inlineCallbacks


configure_logging()


def text_resource(content):
    class Page(Resource):
        isLeaf = True
        def render_GET(self, request):
            request.setHeader(b'content-type', b'text/html')
            return to_bytes(content)
    return Page
