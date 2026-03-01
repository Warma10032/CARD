#!/usr/bin/env python
# -*- coding: utf-8 -*-

import arxiv
from CARD.tools.search.search import Search
from CARD.tools.search.search_registry import SearchRegistry


@SearchRegistry.register("Arxiv")
class ArxivSearch(Search):
    def __init__(self):
        super().__init__()
        self.name = "ArXiv Searcher"
        self.description = "Search for a paper on ArXiv"

    def search(self, query: str) -> str:
        search = arxiv.Search(
            query=query,
            max_results=1,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending,
        )
        results = arxiv.Client().results(search)
        paper = next(results, None)
        if paper:
            return f"Title: {paper.title}\nAuthors: {', '.join(a.name for a in paper.authors)}\nSummary: {paper.summary}"
        return None

    async def search_async(self, query: str) -> str:
        return await search_arxiv(query)

    async def search_batch(self, queries: list[str]) -> list[str]:
        return await search_arxiv_main(queries)


async def search_arxiv(query):
    search = arxiv.Search(
        query=query,
        max_results=2,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )
    results = arxiv.Client().results(search)
    ret = ""
    for paper in results:
        ret += f"Title: {paper.title}\nAuthors: {', '.join(a.name for a in paper.authors)}\nSummary: {paper.summary}\n\n"
    return ret


async def search_arxiv_main(queries):
    tasks = [search_arxiv(query) for query in queries]
    results = await asyncio.gather(*tasks)
    return results
