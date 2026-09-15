import sys
import types

from ml.retrieval.web_shop import build_query, google_shopping_url


def test_build_query_includes_color_and_category():
    assert build_query("olive", "coat", "Solid") == "olive coat buy"


def test_google_shopping_url_encodes_spaces():
    url = google_shopping_url("olive wool coat")
    assert url.startswith("https://www.google.com/search?tbm=shop&q=")
    assert "olive" in url


def test_ddg_search_works_without_shopping_method(monkeypatch):
    """ddgs 9 has no `shopping`; results must come from image and text search."""

    class FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def images(self, query, max_results=5):
            return [{
                "title": "Black tiered skirt",
                "url": "https://shop.example.com/skirt",
                "image": "https://img.example.com/skirt.jpg",
            }]

        def text(self, query, max_results=5):
            return [{"title": "Black skirts", "href": "https://store.example.com/skirts", "body": ""}]

    fake = types.ModuleType("ddgs")
    fake.DDGS = FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", fake)

    from ml.retrieval import web_shop

    products = web_shop._ddg_shopping("black skirt buy", n=2)

    assert [p["url"] for p in products] == [
        "https://shop.example.com/skirt",
        "https://store.example.com/skirts",
    ]
    assert products[0]["imageUrl"] == "https://img.example.com/skirt.jpg"
    assert all(p["source"] == "duckduckgo" for p in products)
