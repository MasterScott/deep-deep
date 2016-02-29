# -*- coding: utf-8 -*-
import re
from urllib.parse import urljoin

from scrapy.linkextractors import IGNORED_EXTENSIONS
from scrapy.utils.url import url_has_any_extension

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
        'before_text': '<text preceeding this link>',
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

        link_text = a.xpath('string()').extract_first(default='')
        img_link_text = a.xpath('./img/@alt').extract_first(default='')
        link['inside_text'] = ' '.join([link_text, img_link_text]).strip()

        # TODO: fix before_text and add after_text
        link['before_text'] = a.xpath('./preceding::text()[1]').extract_first(default='').strip()[-100:]

        yield link
