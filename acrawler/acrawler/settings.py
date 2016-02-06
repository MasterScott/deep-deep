# -*- coding: utf-8 -*-

# Scrapy settings for acrawler project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     http://doc.scrapy.org/en/latest/topics/settings.html
#     http://scrapy.readthedocs.org/en/latest/topics/downloader-middleware.html
#     http://scrapy.readthedocs.org/en/latest/topics/spider-middleware.html

BOT_NAME = 'acrawler'

SPIDER_MODULES = ['acrawler.spiders']
NEWSPIDER_MODULE = 'acrawler.spiders'

# Crawl responsibly by identifying yourself (and your website) on the user-agent
USER_AGENT = 'acrawler'

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Configure maximum concurrent requests performed by Scrapy (default: 16)
CONCURRENT_REQUESTS = 48

# limit crawl to 50K successful pages
CLOSESPIDER_ITEMCOUNT = 50000

REACTOR_THREADPOOL_MAXSIZE = 10

RETRY_ENABLED = False
AJAXCRAWL_ENABLED = True
DOWNLOAD_WARNSIZE = 2*1024*1024
DOWNLOAD_MAXSIZE = 4*1024*1024

# Enable and configure the AutoThrottle extension (disabled by default)
# See http://doc.scrapy.org/en/latest/topics/autothrottle.html
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 0.5
AUTOTHROTTLE_MAX_DELAY = 10
AUTOTHROTTLE_TARGET_CONCURRENCY = 2.0

# Disable cookies for broad crawl (enabled by default)
COOKIES_ENABLED = False


TELNETCONSOLE_ENABLED = True


SCHEDULER = 'acrawler.scheduler.Scheduler'


# Enable and configure HTTP caching (disabled by default)
# See http://scrapy.readthedocs.org/en/latest/topics/downloader-middleware.html#httpcache-middleware-settings
import sys

HTTPCACHE_ENABLED = True
HTTPCACHE_EXPIRATION_SECS = 60*60*24*7
HTTPCACHE_DIR = 'httpcache-%s' % sys.version_info[0]
#HTTPCACHE_IGNORE_HTTP_CODES = []
HTTPCACHE_STORAGE = 'scrapy.extensions.httpcache.FilesystemCacheStorage'
HTTPCACHE_GZIP = False


# Override the default request headers:
#DEFAULT_REQUEST_HEADERS = {
#   'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
#   'Accept-Language': 'en',
#}

# Enable or disable spider middlewares
# See http://scrapy.readthedocs.org/en/latest/topics/spider-middleware.html
#SPIDER_MIDDLEWARES = {
#    'acrawler.middlewares.MyCustomSpiderMiddleware': 543,
#}

# Enable or disable downloader middlewares
# See http://scrapy.readthedocs.org/en/latest/topics/downloader-middleware.html
DOWNLOADER_MIDDLEWARES = {
   'acrawler.middlewares.OffsiteDownloaderMiddleware': 543,
}

# Enable or disable extensions
# See http://scrapy.readthedocs.org/en/latest/topics/extensions.html
#EXTENSIONS = {
#    'scrapy.extensions.telnet.TelnetConsole': None,
#}

# Configure item pipelines
# See http://scrapy.readthedocs.org/en/latest/topics/item-pipeline.html
#ITEM_PIPELINES = {
#    'acrawler.pipelines.SomePipeline': 300,
#}

