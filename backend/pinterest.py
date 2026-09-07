"""Pinterest integration for the Reference page.

Isolated Pinterest logic so the FastAPI layer stays thin:

- auth_url()      -> build the OAuth consent URL
- exchange_code() -> swap an authorization code for a token (stored in memory)
- outfit_feed()   -> return outfit pins for the Reference page, with a curated
                     fallback feed when no token/credentials are configured.

Uses only the stdlib (urllib) to avoid adding dependencies. All failures fall
back gracefully so the Reference page always has something to show.
"""
import base64
import json
import os
import random
import urllib.parse
import urllib.request

PINTEREST_API = "https://api.pinterest.com/v5"
PINTEREST_OAUTH_PAGE = "https://www.pinterest.com/oauth/"

SCOPES = "pins:read,boards:read,user_accounts:read"

# Curated fallback feed (real, hotlinkable outfit photos already used in the app).
# Used when Pinterest credentials/token are not configured, so the feature works
# out of the box before keys are added.
FALLBACK_OUTFITS = [
    {"id": "fb_1", "imageUrl": "https://images.pexels.com/photos/27641316/pexels-photo-27641316.jpeg?auto=compress&cs=tinysrgb&w=600", "title": "Casual Chic"},
    {"id": "fb_2", "imageUrl": "https://images.pexels.com/photos/31046830/pexels-photo-31046830.jpeg?auto=compress&cs=tinysrgb&w=600", "title": "Minimalist"},
    {"id": "fb_3", "imageUrl": "https://images.pexels.com/photos/29398132/pexels-photo-29398132.jpeg?auto=compress&cs=tinysrgb&w=600", "title": "Y2K"},
    {"id": "fb_4", "imageUrl": "https://images.pexels.com/photos/31046829/pexels-photo-31046829.jpeg?auto=compress&cs=tinysrgb&w=600", "title": "Streetwear"},
    {"id": "fb_5", "imageUrl": "https://images.pexels.com/photos/31046827/pexels-photo-31046827.jpeg?auto=compress&cs=tinysrgb&w=600", "title": "Preppy"},
    {"id": "fb_6", "imageUrl": "https://images.pexels.com/photos/30381008/pexels-photo-30381008.jpeg?auto=compress&cs=tinysrgb&w=600", "title": "Vintage"},
    {"id": "fb_7", "imageUrl": "https://images.pexels.com/photos/31046841/pexels-photo-31046841.jpeg?auto=compress&cs=tinysrgb&w=600", "title": "Boho"},
    {"id": "fb_8", "imageUrl": "https://images.pexels.com/photos/30590661/pexels-photo-30590661.jpeg?auto=compress&cs=tinysrgb&w=600", "title": "Elegant"},
    {"id": "fb_9", "imageUrl": "https://images.pexels.com/photos/13568592/pexels-photo-13568592.jpeg?auto=compress&cs=tinysrgb&w=600", "title": "Business Casual"},
    {"id": "fb_10", "imageUrl": "https://images.pexels.com/photos/13568611/pexels-photo-13568611.jpeg?auto=compress&cs=tinysrgb&w=600", "title": "Edgy"},
    {"id": "fb_11", "imageUrl": "https://images.pexels.com/photos/27542890/pexels-photo-27542890.jpeg?auto=compress&cs=tinysrgb&w=600", "title": "Sporty"},
    {"id": "fb_12", "imageUrl": "https://images.pexels.com/photos/27383816/pexels-photo-27383816.jpeg?auto=compress&cs=tinysrgb&w=600", "title": "Formal"},
]

# Outfit search terms rotated through when the real API is available.
SEARCH_QUERIES = ["outfit", "street style", "minimalist outfit", "summer outfit", "work outfit"]


class PinterestClient:
    """Small stateful wrapper holding the OAuth token in memory for the session."""

    def __init__(self):
        self._token = os.getenv("PINTEREST_ACCESS_TOKEN", "").strip() or None

    @property
    def client_id(self) -> str:
        return os.getenv("PINTEREST_CLIENT_ID", "").strip()

    @property
    def client_secret(self) -> str:
        return os.getenv("PINTEREST_CLIENT_SECRET", "").strip()

    @property
    def redirect_uri(self) -> str:
        return os.getenv("PINTEREST_REDIRECT_URI", "http://localhost:8000/api/pinterest/callback").strip()

    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    @property
    def token(self) -> str | None:
        return self._token

    def auth_url(self, state: str = "styla") -> str | None:
        if not self.configured:
            return None
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": SCOPES,
            "state": state,
        }
        return f"{PINTEREST_OAUTH_PAGE}?{urllib.parse.urlencode(params)}"

    def exchange_code(self, code: str) -> bool:
        if not self.configured:
            return False
        credentials = f"{self.client_id}:{self.client_secret}"
        auth = base64.b64encode(credentials.encode()).decode()
        data = urllib.parse.urlencode(
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self.redirect_uri,
            }
        ).encode()
        req = urllib.request.Request(
            f"{PINTEREST_API}/oauth/token",
            data=data,
            headers={
                "Authorization": f"Basic {auth}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                payload = json.loads(resp.read().decode())
            token = payload.get("access_token")
            if token:
                self._token = token
                return True
        except Exception:
            pass
        return False

    def outfit_feed(self, page_size: int = 25) -> list[dict]:
        if self._token:
            pins = self._search_pins(page_size)
            if pins:
                return pins
        return list(FALLBACK_OUTFITS)

    def _search_pins(self, page_size: int) -> list[dict]:
        query = random.choice(SEARCH_QUERIES)
        url = (
            f"{PINTEREST_API}/search/pins?"
            f"query={urllib.parse.quote(query)}&page_size={page_size}"
        )
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                payload = json.loads(resp.read().decode())
        except Exception:
            return []

        items = payload.get("items") or []
        pins = []
        for pin in items:
            image_url = _pin_image_url(pin)
            if not image_url:
                continue
            pins.append(
                {
                    "id": pin.get("id", image_url),
                    "imageUrl": image_url,
                    "title": pin.get("title") or pin.get("note") or "Outfit",
                    "link": pin.get("link"),
                }
            )
        return pins


def _pin_image_url(pin: dict) -> str | None:
    media = pin.get("media") or {}
    images = media.get("images") or {}
    for key in ("originals", "1200x", "600x", "400x300", "150x150"):
        img = images.get(key) or {}
        url = img.get("url")
        if url:
            return url
    return None


_client: PinterestClient | None = None


def get_client() -> PinterestClient:
    global _client
    if _client is None:
        _client = PinterestClient()
    return _client
