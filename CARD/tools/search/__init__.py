#!/usr/bin/env python
# -*- coding: utf-8 -*-

from CARD.tools.search.search import Search
from CARD.tools.search.search_registry import SearchRegistry
from CARD.tools.search.arXiv import ArxivSearch
from CARD.tools.search.google import GoogleSearch
from CARD.tools.search.baidu import BaiduSearch
from CARD.tools.search.duckduckgo import DuckDuckGoSearch
from CARD.tools.search.wiki import WikiSearch

__all__ = [
    "Search",
    "SearchRegistry",
    "ArxivSearch",
    "GoogleSearch",
    "BaiduSearch",
    "DuckDuckGoSearch",
    "WikiSearch",
]
