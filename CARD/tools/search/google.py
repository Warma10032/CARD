#!/usr/bin/env python
# -*- coding: utf-8 -*-

from googlesearch import search
import asyncio
import random
import requests
from bs4 import BeautifulSoup
from CARD.tools.search.search import Search
from CARD.tools.search.search_registry import SearchRegistry


@SearchRegistry.register("Google")
class GoogleSearch(Search):
    def __init__(self):
        super().__init__()
        self.name = "Google SearchEngine"
        self.description = "Search for an item in Google"

    async def search(self, query: str, site: str = None) -> str:
        try:
            if site:
                site_query = self.search_sites.get(site.lower(), "")
                query = f"{site_query} {query}"

            result = list(search(query[:300], advanced=True, num_results=1))
            if len(result) > 0:
                content = await self._get_page_content(result.url)
                return f"Search: {query}, get: {result.title}/n {result.description}/n {content}"
            return ""
        except Exception as e:
            print(f"Google search error: {e}")
            return ""

    async def _get_page_content(self, url):
        try:
            response = requests.get(
                url, headers={"User-Agent": "Mozilla/5.0", "Accept-Charset": "utf-8"}
            )
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                # Extract main content
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
                    results = list(
                        search(full_query[:300], advanced=True, num_results=2)
                    )
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
            print(f"Google search error: {e}")
            return ""

    async def _get_page_summary(self, result, query: str = None):
        try:
            response = requests.get(
                result.url,
                headers={"User-Agent": "Mozilla/5.0", "Accept-Charset": "utf-8"},
            )
            if response.status_code == 200:
                if response.encoding.lower() not in ["utf-8", "utf8"]:
                    return f"{result.title}\n{result.description}"
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

                # Ensure total content length is within reasonable range
                summary = "\n".join(valid_contents)

                return f"{result.title}\n{result.description}\n{summary}"
            return ""
        except Exception as e:
            print(f"Failed to get page summary: {e}")
            return ""


if __name__ == "__main__":
    search_google = GoogleSearch()
    queries = ["Python", "Asyncio", "LLM"]
    result = asyncio.run(search_google.search_batch(queries))
    print(result)
