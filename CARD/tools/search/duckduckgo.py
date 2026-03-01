#!/usr/bin/env python
# -*- coding: utf-8 -*-

from duckduckgo_search import DDGS
import asyncio
import requests
import random
from bs4 import BeautifulSoup
from CARD.tools.search.search import Search
from CARD.tools.search.search_registry import SearchRegistry


@SearchRegistry.register("DuckDuckGo")
class DuckDuckGoSearch(Search):
    def __init__(self):
        super().__init__()
        self.name = "DuckDuckGo SearchEngine"
        self.description = "Search for text in DuckDuckGo"

    async def search(self, query: str, site: str = None) -> str:
        try:
            if site:
                site_query = self.search_sites.get(site.lower(), "")
                query = f"{site_query} {query}"

            with DDGS() as ddgs:
                results = list(ddgs.text(query[:300], max_results=1))

            if results and len(results) > 0:
                content = await self._get_page_content(results[0]["href"])
                return f'Search:{query}, get:{results[0]["title"]}\n {results[0]["body"]}\n {content}'
            return ""
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return ""

    async def _get_page_content(self, url: str) -> str:
        try:
            response = requests.get(
                url, headers={"User-Agent": "Mozilla/5.0", "Accept-Charset": "utf-8"}
            )
            if response.status_code == 200:
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
                full_query = f"{site_query} {query}"
            else:
                full_query = query

            # Increase request interval and retry mechanism
            max_retries = 3
            retry_delay = 1  # Initial delay 1 second
            results = []

            for attempt in range(max_retries):
                try:
                    await asyncio.sleep(
                        retry_delay + random.uniform(0, 2)
                    )  # Random wait 1-3 seconds
                    with DDGS() as ddgs:
                        results = list(ddgs.text(full_query[:300], max_results=2))
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        retry_delay *= 2  # Exponential backoff
                        continue
                    raise

            tasks = [self._get_page_summary(result, query) for result in results]
            summaries = await asyncio.gather(*tasks)

            return f"Search:{query}, get:{summaries}"
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return ""

    async def _get_page_summary(self, result: dict, query: str = None) -> str:
        try:
            response = requests.get(
                result["href"],
                headers={"User-Agent": "Mozilla/5.0", "Accept-Charset": "utf-8"},
            )
            if response.status_code == 200:
                if response.encoding.lower() not in ["utf-8", "utf8"]:
                    return f'{result["title"]}\n{result["body"]}'

                soup = BeautifulSoup(response.text, "html.parser")

                # Get all tags that may contain main content
                content_tags = (
                    soup.find_all(["h1", "h2", "h3"])
                    + soup.find_all("p")
                    + soup.find_all("article")
                    + soup.find_all(class_=["content", "article", "post-content"])
                )

                # Filter and process paragraphs
                valid_contents = []
                keywords = query.lower().split() if query else []

                for tag in content_tags:
                    if len(valid_contents) >= 3:
                        break

                    text = tag.get_text().strip()
                    if len(text) < 50:
                        continue

                    # Check keyword matching
                    text_lower = text.lower()
                    if keywords and not any(
                        keyword in text_lower for keyword in keywords
                    ):
                        continue

                    # Limit max length of single paragraph
                    if len(text) > 300:
                        text = text[:300] + "..."
                    # Remove extra whitespace
                    text = " ".join(text.split())
                    valid_contents.append(text)

                summary = "\n".join(valid_contents)
                return f'{result["title"]}\n{result["body"]}\n{summary}'
            return ""
        except Exception as e:
            print(f"Failed to get page summary: {e}")
            return ""


if __name__ == "__main__":
    search_ddg = DuckDuckGoSearch()
    queries = ["Python", "Asyncio", "LLM"]
    result = asyncio.run(search_ddg.search_batch(queries, "reddit"))
    print(result)
