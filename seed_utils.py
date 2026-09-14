"""Shared helper for the seed scripts: sign in and return auth headers."""
import sys

import requests


def login(api: str, email: str, password: str) -> dict:
    """Sign in (creating the account if needed) and return Authorization headers."""
    api = api.rstrip("/")
    res = requests.post(f"{api}/api/auth/login", json={"email": email, "password": password}, timeout=30)
    if res.status_code == 401:
        res = requests.post(
            f"{api}/api/auth/signup",
            json={"email": email, "password": password, "name": email.split("@")[0]},
            timeout=30,
        )
    if res.status_code != 200:
        print(f"Could not sign in as {email}: {res.status_code} {res.text}")
        sys.exit(1)
    print(f"Signed in as {email}")
    return {"Authorization": f"Bearer {res.json()['token']}"}
