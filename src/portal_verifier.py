from __future__ import annotations

from dataclasses import dataclass
from typing import List
from urllib.parse import quote_plus

import feedparser
import requests

from src.preprocess import extract_keywords


OFFICIAL_DOMAINS = ["bbc.com", "reuters.com", "thehindu.com", "ndtv.com"]


@dataclass
class OfficialArticle:
    title: str
    summary: str
    link: str
    source_domain: str

    @property
    def combined_text(self) -> str:
        return f"{self.title} {self.summary}".strip()


def _build_google_news_query(news_text: str) -> str:
    keywords = extract_keywords(news_text, top_k=6)
    term = " ".join(keywords) if keywords else news_text[:120]
    site_filter = " OR ".join(f"site:{domain}" for domain in OFFICIAL_DOMAINS)
    return f"{term} ({site_filter})"


def _google_news_rss_url(query: str) -> str:
    encoded_query = quote_plus(query)
    return f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"


def fetch_official_articles(news_text: str, timeout: int = 8, limit: int = 12) -> List[OfficialArticle]:
    query = _build_google_news_query(news_text)
    rss_url = _google_news_rss_url(query)

    try:
        response = requests.get(rss_url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException:
        return []

    parsed = feedparser.parse(response.content)
    results: List[OfficialArticle] = []
    for entry in parsed.entries[:limit]:
        link = getattr(entry, "link", "") or ""
        title = getattr(entry, "title", "") or ""
        summary = getattr(entry, "summary", "") or ""
        source_domain = ""
        if getattr(entry, "source", None):
            source_domain = (entry.source.get("href") or entry.source.get("title") or "").lower()
        results.append(
            OfficialArticle(
                title=title,
                summary=summary,
                link=link,
                source_domain=source_domain,
            )
        )
    return results

