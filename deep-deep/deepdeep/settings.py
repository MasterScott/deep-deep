# -*- coding: utf-8 -*-

# Scrapy settings for deepdeep project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     http://doc.scrapy.org/en/latest/topics/settings.html
#     http://scrapy.readthedocs.org/en/latest/topics/downloader-middleware.html
#     http://scrapy.readthedocs.org/en/latest/topics/spider-middleware.html

BOT_NAME = 'deepdeep'

SPIDER_MODULES = ['deepdeep.spiders']
NEWSPIDER_MODULE = 'deepdeep.spiders'

# Crawl responsibly by identifying yourself (and your website) on the user-agent
USER_AGENT = 'deepdeep'

# Don't obey robots.txt rules
ROBOTSTXT_OBEY = False

# Configure maximum concurrent requests performed by Scrapy (default: 16)
CONCURRENT_REQUESTS = 48

# limit crawl to 50K successful pages
CLOSESPIDER_ITEMCOUNT = 50000

REACTOR_THREADPOOL_MAXSIZE = 10

RETRY_ENABLED = False
AJAXCRAWL_ENABLED = True
DOWNLOAD_WARNSIZE = 1*1024*1024
DOWNLOAD_MAXSIZE = 1*1024*1024
DOWNLOAD_TIMEOUT = 60

# use very large priority adjust values to make sure redirects are processed
REDIRECT_PRIORITY_ADJUST = 100*10000

# Enable and configure the AutoThrottle extension (disabled by default)
# See http://doc.scrapy.org/en/latest/topics/autothrottle.html
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 0.5
AUTOTHROTTLE_MAX_DELAY = 5
AUTOTHROTTLE_TARGET_CONCURRENCY = 2.0

# Disable cookies for broad crawl (enabled by default)
COOKIES_ENABLED = False

MEMUSAGE_ENABLED = True
TELNETCONSOLE_ENABLED = True  # it doesn't work in Python 3 yet!
MONITOR_DOWNLOADS_INTERVAL = 10
DUMP_STATS_INTERVAL = 30

SCHEDULER = 'deepdeep.scheduler.Scheduler'


# Enable and configure HTTP caching (disabled by default)
# See http://scrapy.readthedocs.org/en/latest/topics/downloader-middleware.html#httpcache-middleware-settings
import sys

HTTPCACHE_ENABLED = False
HTTPCACHE_EXPIRATION_SECS = 60*60*24*30  # 30 days
HTTPCACHE_DIR = 'httpcache-%s' % sys.version_info[0]
#HTTPCACHE_IGNORE_HTTP_CODES = []
HTTPCACHE_STORAGE = 'scrapy.extensions.httpcache.FilesystemCacheStorage'
HTTPCACHE_GZIP = False


FEED_STORAGES = {
    'gzip': 'deepdeep.exports.GzipFileFeedStorage',
}


# Override the default request headers:
#DEFAULT_REQUEST_HEADERS = {
#   'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
#   'Accept-Language': 'en',
#}

# Enable or disable spider middlewares
# See http://scrapy.readthedocs.org/en/latest/topics/spider-middleware.html
SPIDER_MIDDLEWARES = {
    'deepdeep.spidermiddlewares.CrawlGraphMiddleware': 400,
}
CRAWLGRAPH_ENABLED = False

# Enable or disable downloader middlewares
# See http://scrapy.readthedocs.org/en/latest/topics/downloader-middleware.html
DOWNLOADER_MIDDLEWARES = {
   'deepdeep.downloadermiddlewares.OffsiteDownloaderMiddleware': 543,
}

# Enable or disable extensions
# See http://scrapy.readthedocs.org/en/latest/topics/extensions.html
EXTENSIONS = {
    'deepdeep.extensions.MonitorDownloadsExtension': 100,
    'deepdeep.extensions.DumpStatsExtension': 101,
}

# Configure item pipelines
# See http://scrapy.readthedocs.org/en/latest/topics/item-pipeline.html
#ITEM_PIPELINES = {
#    'deepdeep.pipelines.SomePipeline': 300,
#}

