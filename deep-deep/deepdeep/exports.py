# -*- coding: utf-8 -*-
import os
import gzip

from zope.interface import Interface, implementer
from w3lib.url import file_uri_to_path
from scrapy.extensions.feedexport import IFeedStorage


@implementer(IFeedStorage)
class GzipFileFeedStorage(object):

    def __init__(self, uri):
        self.path = file_uri_to_path(uri) + ".gz"

    def open(self, spider):
        dirname = os.path.dirname(self.path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        return gzip.open(self.path, 'ab', compresslevel=4, )

    def store(self, file):
        file.close()
