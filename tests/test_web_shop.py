from ml.retrieval.web_shop import build_query, google_shopping_url


def test_build_query_includes_color_and_category():
    assert build_query("olive", "coat", "Solid") == "olive coat buy"


def test_google_shopping_url_encodes_spaces():
    url = google_shopping_url("olive wool coat")
    assert url.startswith("https://www.google.com/search?tbm=shop&q=")
    assert "olive" in url
