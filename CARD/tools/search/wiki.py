#!/usr/bin/env python
# -*- coding: utf-8 -*-

import wikipedia
import asyncio
from CARD.tools.search.search import Search
from CARD.tools.search.search_registry import SearchRegistry


@SearchRegistry.register("Wiki")
class WikiSearch(Search):
    def __init__(self):
        super().__init__()
        self.name = "Wikipedia SearchEngine"
        self.description = "Seach for an item in Wikipedia"

    def search(self, query: str) -> str:
        result = wikipedia.search(query[:300], results=1, suggestion=True)
        print(result)
        if len(result[0]) != 0:
            return wikipedia.page(title=result[0]).content

        if result[1] is not None:
            result = wikipedia.search(result[1], results=1)
            return wikipedia.page(title=result[0]).content

        return None

    async def search_async(self, query: str, site: str = None) -> str:
        return await self.search_wiki(query)

    async def search_batch(self, queries: list[str], site: str = None) -> list[str]:
        tasks = [self.search_wiki(query) for query in queries]
        return await asyncio.gather(*tasks)

    async def get_wikipedia_summary(self, title):
        try:
            wikipedia.set_lang("en")
            summ = wikipedia.summary(title)
            return summ
        except wikipedia.exceptions.DisambiguationError as e:
            return await self.get_wikipedia_summary(e.options[0])
        except wikipedia.exceptions.PageError:
            return ""

    async def search_wiki(self, query):
        wikipedia.set_lang("en")
        result = wikipedia.search(query, results=1, suggestion=True)
        ret = ""
        tasks = []

        if len(result[0]) != 0:
            for res in result[0]:
                tasks.append(self.get_wikipedia_summary(res))
            summaries = await asyncio.gather(*tasks)
            for res, summa in zip(result[0], summaries):
                if len(summa):
                    ret += f"The summary of {res} in Wikipedia is: {summa}\n"
        if result[1] is not None:
            summa = await self.get_wikipedia_summary(result[1])
            if len(summa):
                ret += f"The summary of {result[1]} in Wikipedia is: {summa}\n"
        return ret


if __name__ == "__main__":
    search_wiki = WikiSearch()
    queries = ["Python", "Asyncio", "LLM"]
    result = asyncio.run(search_wiki.search_batch(queries))
    print(result)
