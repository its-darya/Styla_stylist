"""Find a garment on the public web and return store pages.

Prefers Google Programmable Search when GOOGLE_API_KEY + GOOGLE_CSE_ID are set.
Falls back to DuckDuckGo shopping/text results so the feature works without keys.
If both fail, returns a Google Shopping search URL so the user still lands on Google.
"""
from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import URLError
from urllib.parse import quote_plus, urlencode
from urllib.request import Request, urlopen


SHOP_HOST_SKIP = (
    "pinterest.",
    "instagram.",
    "tiktok.",
    "facebook.",
    "youtube.",
    "reddit.",
)


def build_query(color: str | None, category: str | None, pattern: str | None = None) -> str:
    parts: list[str] = []
    color = (color or "").strip()
    category = (category or "").strip()
    pattern = (pattern or "").strip()
    if color and color.lower() not in {"unknown", "multi-color"}:
        parts.append(color)
    if pattern and pattern.lower() not in {"solid", "unknown"}:
        parts.append(pattern)
    if category:
        parts.append(category)
    parts.append("buy")
    return " ".join(parts).strip() or "clothing buy"


def google_shopping_url(query: str) -> str:
    return f"https://www.google.com/search?tbm=shop&q={quote_plus(query)}"


def _is_shop_url(url: str) -> bool:
    low = url.lower()
    if not low.startswith("http"):
        return False
    return not any(host in low for host in SHOP_HOST_SKIP)


def _normalize_product(
    *,
    name: str,
    url: str,
    image_url: str = "",
    price: str = "",
    source: str = "",
) -> dict[str, str] | None:
    if not url or not _is_shop_url(url):
        return None
    return {
        "name": (name or "Similar item").strip()[:180],
        "url": url,
        "imageUrl": image_url,
        "price": price or "",
        "source": source,
    }


def _google_cse(query: str, n: int) -> list[dict[str, str]]:
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    cx = os.getenv("GOOGLE_CSE_ID", "").strip()
    if not key or not cx:
        return []
    params = urlencode(
        {
            "key": key,
            "cx": cx,
            "q": query,
            "num": min(max(n, 1), 10),
            "safe": "active",
        }
    )
    req = Request(
        f"https://www.googleapis.com/customsearch/v1?{params}",
        headers={"User-Agent": "Styla/1.0"},
    )
    try:
        with urlopen(req, timeout=12) as resp:
            payload: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
    except (URLError, TimeoutError, json.JSONDecodeError, OSError):
        return []

    products: list[dict[str, str]] = []
    for item in payload.get("items") or []:
        image = ""
        pagemap = item.get("pagemap") or {}
        cse_images = pagemap.get("cse_image") or pagemap.get("cse_thumbnail") or []
        if cse_images and isinstance(cse_images, list):
            image = str(cse_images[0].get("src") or "")
        product = _normalize_product(
            name=str(item.get("title") or query),
            url=str(item.get("link") or ""),
            image_url=image,
            source="google",
        )
        if product:
            products.append(product)
        if len(products) >= n:
            break
    return products


def _ddg_shopping(query: str, n: int) -> list[dict[str, str]]:
    try:
        from ddgs import DDGS
    except ImportError:
        return []

    raw: list[dict[str, Any]] = []
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.shopping(query, max_results=n + 4) or [])
            if len(raw) < n:
                raw.extend(list(ddgs.text(query, max_results=n) or []))
    except Exception:
        return []

    products: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in raw:
        url = str(row.get("url") or row.get("href") or row.get("link") or "")
        if url in seen:
            continue
        product = _normalize_product(
            name=str(row.get("title") or query),
            url=url,
            image_url=str(row.get("image") or row.get("thumbnail") or ""),
            price=str(row.get("price") or ""),
            source="duckduckgo",
        )
        if not product:
            continue
        seen.add(url)
        products.append(product)
        if len(products) >= n:
            break
    return products


def search_similar_products(
    color: str | None,
    category: str | None,
    pattern: str | None = None,
    n: int = 3,
) -> tuple[str, str, list[dict[str, str]]]:
    """Return (query, google_shopping_url, products)."""
    query = build_query(color, category, pattern)
    shopping_url = google_shopping_url(query)
    products = _google_cse(query, n)
    if not products:
        products = _ddg_shopping(query, n)
    if not products:
        products = [
            {
                "name": f"Google Shopping: {query}",
                "url": shopping_url,
                "imageUrl": "",
                "price": "",
                "source": "google-shopping",
            }
        ]
    return query, shopping_url, products[:n]
