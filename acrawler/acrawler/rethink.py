# -*- coding: utf-8 -*-
"""
RethinkDB-backed requests queue.

XXX: This doesn't work. Think again about using RethinkDB;
it doesn't have basic features like filtering by one index and then
sorting using another index.
"""
import contextlib
import rethinkdb as r

from scrapy.utils.python import to_unicode
from scrapy.utils.reqser import request_from_dict, request_to_dict


@contextlib.contextmanager
def ignore_exception(exc):
    try:
        yield
    except exc:
        pass


class RethinkDBRequestQueue(object):
    STATUS_NEW = 'new'
    STATUS_PROGRESS = 'progress'
    STATUS_DONE = 'done'

    def __init__(self, spider, crawl_id, db='acrawler', table='requests'):
        self.spider = spider
        self.crawl_id = crawl_id
        self.conn = r.connect(db=db)
        self.requests = r.table(table)

    def close(self):
        self.conn.close()

    @classmethod
    def init_tables(cls, db='acrawler', table='requests'):
        with r.connect(db=db) as c:
            with ignore_exception(r.ReqlRuntimeError):
                r.table_create(table).run(c)

            requests = r.table(table)
            for index in ['priority']:
                with ignore_exception(r.ReqlRuntimeError):
                    requests.index_create(index).run(c)

            with ignore_exception(r.ReqlRuntimeError):
                requests.\
                    index_create(
                        'crawl_id_and_status',
                        [r.row['crawl_id'], r.row['status']]
                    )\
                    .run(c)

    def push(self, request):
        row = self._serialize_request(request)
        self.requests.insert(row).run(self.conn)
        return True

    def pop(self):
        rows = self.requests\
            .get_all([self.crawl_id, self.STATUS_NEW], index='crawl_id_and_status')\
            .order_by(index=r.desc('priority'))\
            .limit(1)\
            .run(self.conn)
        rows = list(rows)
        if not rows:
            return
        req = rows[0]
        self.requests\
            .get(req['id'])\
            .update({'status': self.STATUS_PROGRESS})\
            .run(self.conn)
        return request_from_dict(req, spider=self.spider)

    # def ack(self, request):
    #     pass

    def __len__(self):
        return 100
        # return 0
        # return len(self.requests)

    def _serialize_request(self, request):
        row = request_to_dict(request, spider=self.spider)
        row['headers'] = {to_unicode(k): v for k,v in row['headers'].items()}
        row['crawl_id'] = self.crawl_id
        row['status'] = self.STATUS_NEW
        return row
