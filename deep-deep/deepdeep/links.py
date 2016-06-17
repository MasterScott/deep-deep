# -*- coding: utf-8 -*-
import re
from urllib.parse import urljoin

from formasaurus.utils import get_domain
from scrapy.linkextractors import IGNORED_EXTENSIONS
from scrapy.utils.response import get_base_url
from scrapy.utils.url import url_has_any_extension
from deepdeep.utils import canonicalize_url

_NEW_IGNORED = {'7z', '7zip', 'xz', 'gz', 'tar', 'bz2', 'cdr', 'apk'}
_IGNORED = set(IGNORED_EXTENSIONS) | _NEW_IGNORED
_IGNORED = {'.' + e for e in _IGNORED}


js_link_search = re.compile(r"(javascript:)?location\.href=['\"](?P<url>.+)['\"]").search
def extract_js_link(href):
    """
    >>> extract_js_link("javascript:location.href='http://www.facebook.com/rivervalleyvet';")
    'http://www.facebook.com/rivervalleyvet'
    >>> extract_js_link("location.href='http://www.facebook.com/rivervalleyvet';")
    'http://www.facebook.com/rivervalleyvet'
    >>> extract_js_link("javascript:href='http://www.facebook.com/rivervalleyvet';") is None
    True
    """
    m = js_link_search(href)
    if m:
        return m.group('url')


def extract_link_dicts(selector, base_url):
    """
    Extract dicts with link information::

    {
        'url': '<absolute URL>',
        'attrs': {
            '<attribute name>': '<value>',
            ...
        },
        'inside_text': '<text inside link>',
        # 'before_text': '<text preceeding this link>',
    }
    """
    selector.remove_namespaces()

    for a in selector.xpath('//a'):
        link = {}

        attrs = a.root.attrib
        if 'href' not in attrs:
            continue

        href = attrs['href']
        if 'mailto:' in href:
            continue

        js_link = extract_js_link(href)
        if js_link:
            href = js_link
            link['js'] = True

        url = urljoin(base_url, href)
        if url_has_any_extension(url, _IGNORED):
            continue

        link['url'] = url
        link['attrs'] = dict(attrs)

        link_text = a.xpath('normalize-space()').extract_first(default='')
        img_link_text = a.xpath('./img/@alt').extract_first(default='')
        link['inside_text'] = ' '.join([link_text, img_link_text]).strip()

        # TODO: fix before_text and add after_text
        # link['before_text'] = a.xpath('./preceding::text()[1]').extract_first(default='').strip()[-100:]

        yield link


def iter_response_link_dicts(response, domain=None):
    base_url = get_base_url(response)
    for link in extract_link_dicts(response.selector, base_url):
        link['domain'] = get_domain(link['url'])
        if domain is not None and link['domain'] != domain:
            continue
        yield link


class DictLinkExtractor:
    """
    A custom link extractor. It returns link dicts instead of Link objects.
    DictLinkExtractor is not compatible with Scrapy link extractors.
    """
    def __init__(self):
        self.seen_urls = set()

    def iter_link_dicts(self, response, domain=None, deduplicate=True,
                        deduplicate_local=True):
        """
        Extract links from the response.
        If ``domain`` is not None, only links for a given domain are returned.
        If ``deduplicate`` is True (default), links with seen URLs
        are not returned.
        If ``deduplicate_local`` is True (default), links which are duplicate
        on a page are not returned.
        """
        links = iter_response_link_dicts(response, domain)
        if deduplicate:
            links = self.deduplicate_links(links)
        elif deduplicate_local:
            links = self.deduplicate_links(links, seen_urls=set())
        return links

    def deduplicate_links(self, links, indices=False, seen_urls=None):
        """
        Filter out links with duplicate URLs.
        Requests are also filtered out in Scheduler by dupefilter.
        Here we filter them to avoid creating unnecessary requests
        in first place; it helps other components like CrawlGraphMiddleware.
        """
        if seen_urls is None:
            seen_urls = self.seen_urls
        for idx, link in enumerate(links):
            url = link['url']
            canonical = canonicalize_url(url)
            if canonical in seen_urls:
                continue
            seen_urls.add(canonical)
            if indices:
                yield idx, link
            else:
                yield link
