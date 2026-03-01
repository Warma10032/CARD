#!/usr/bin/env python
# -*- coding: utf-8 -*-

from baidusearch.baidusearch import search
import asyncio
import requests
from bs4 import BeautifulSoup
from CARD.tools.search.search import Search
from CARD.tools.search.search_registry import SearchRegistry


@SearchRegistry.register("Baidu")
class BaiduSearch(Search):
    def __init__(self):
        super().__init__()
        self.name = "Baidu SearchEngine"
        self.description = "Search for an item in Baidu"

    async def search(self, query: str, site: str = None) -> str:
        try:
            if site:
                site_query = self.search_sites.get(site.lower(), "")
                query = f"{site_query} {query}"

            results = list(search(query[:300], num_results=1))
            if len(results) > 0:
                result = results[0]
                content = await self._get_page_content(result["url"])
                return f"Search:{query}, get:{' '.join(result['title'].split())}\n{' '.join(result['abstract'].split())}\n {content}"
            return ""
        except Exception as e:
            print(f"Baidu search error: {e}")
            return ""

    async def _get_page_content(self, url):
        try:
            response = requests.get(
                url, headers={"User-Agent": "Mozilla/5.0", "Accept-Charset": "utf-8"}
            )
            if response.status_code == 200:
                if response.encoding.lower() not in ["utf-8", "utf8"]:
                    return ""
                soup = BeautifulSoup(response.text, "html.parser")
                paragraphs = soup.find_all("p")
                content = "\n".join([p.get_text() for p in paragraphs])
                return content
            return ""
        except Exception as e:
            print(f"Failed to get page content: {e}")
            return ""

    async def search_async(self, query: str, site: str = None) -> str:
        return await self.search_summary(query, site)

    async def search_batch(self, queries: list[str], site: str = None) -> list[str]:
        tasks = [self.search_summary(query, site) for query in queries]
        return await asyncio.gather(*tasks)

    async def search_summary(self, query: str, site: str = None) -> str:
        try:
            if site:
                site_query = self.search_sites.get(site.lower(), "")
                query = f"{site_query} {query}"

            results = list(search(query[:300], num_results=2))
            tasks = [self._get_page_summary(result, query) for result in results]
            summaries = await asyncio.gather(*tasks)

            return f"Search:{query}, get:{summaries}"
        except Exception as e:
            print(f"Baidu search error: {e}")
            return ""

    async def _get_page_summary(self, result, query: str = None):
        try:
            response = requests.get(
                result["url"],
                headers={"User-Agent": "Mozilla/5.0", "Accept-Charset": "utf-8"},
            )
            if response.status_code == 200:
                if response.encoding.lower() not in ["utf-8", "utf8"]:
                    return f"{' '.join(result['title'].split())}\n{' '.join(result['abstract'].split())}"

                soup = BeautifulSoup(response.text, "html.parser")
                content_tags = (
                    soup.find_all(["h1", "h2", "h3"])
                    + soup.find_all("p")
                    + soup.find_all("article")
                    + soup.find_all(class_=["content", "article", "post-content"])
                )

                valid_contents = []
                keywords = query.lower().split() if query else []

                for tag in content_tags:
                    if len(valid_contents) >= 3:
                        break

                    text = tag.get_text().strip()
                    if len(text) < 50:
                        continue

                    text_lower = text.lower()
                    if keywords and not any(
                        keyword in text_lower for keyword in keywords
                    ):
                        continue

                    if len(text) > 300:
                        text = text[:300] + "..."
                    text = " ".join(text.split())
                    valid_contents.append(text)

                summary = "\n".join(valid_contents)
                return f"{' '.join(result['title'].split())}\n{' '.join(result['abstract'].split())}\n{summary}"
            return ""
        except Exception as e:
            print(f"Failed to get page summary: {e}")
            return ""


if __name__ == "__main__":
    search_baidu = BaiduSearch()
    queries = ["Python", "Asyncio", "LLM"]
    result = asyncio.run(search_baidu.search_batch(queries))
    print(result)
