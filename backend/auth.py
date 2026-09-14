"""Accounts: sign up, sign in, and the `current_user` dependency.

- Passwords are hashed with bcrypt (never stored in clear).
- Sessions are stateless JWTs (HS256) signed with `STYLA_JWT_SECRET`. When
  the variable is unset a random secret is generated once and kept in
  `data/.jwt_secret`, so tokens survive backend restarts in development.
- Every protected endpoint declares `user: CurrentUser = Depends(current_user)`
  and scopes its queries by `user.id`.
"""
from __future__ import annotations

import os
import re
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import bcrypt
import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from backend.db import Database

BASE_DIR = Path(__file__).resolve().parent.parent
TOKEN_TTL_DAYS = int(os.getenv("STYLA_TOKEN_TTL_DAYS", "7"))
JWT_ALGORITHM = "HS256"
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ---------------------------------------------------------------------------
# Secrets & hashing
# ---------------------------------------------------------------------------

def _jwt_secret() -> str:
    secret = os.getenv("STYLA_JWT_SECRET", "").strip()
    if secret:
        return secret
    secret_file = BASE_DIR / "data" / ".jwt_secret"
    if secret_file.exists():
        return secret_file.read_text(encoding="utf-8").strip()
    secret_file.parent.mkdir(parents=True, exist_ok=True)
    secret = secrets.token_urlsafe(48)
    secret_file.write_text(secret, encoding="utf-8")
    print("[auth] STYLA_JWT_SECRET not set — generated one in data/.jwt_secret")
    return secret


JWT_SECRET = _jwt_secret()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("ascii"))
    except ValueError:
        return False


def create_token(user_id: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {"sub": user_id, "iat": now, "exp": now + timedelta(days=TOKEN_TTL_DAYS)}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Optional[str]:
    """Return the user id inside a valid token, else None."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None
    sub = payload.get("sub")
    return str(sub) if sub else None


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CurrentUser:
    id: str
    email: str
    name: str

    def public(self) -> dict:
        return {"id": self.id, "email": self.email, "name": self.name}


class SignupRequest(BaseModel):
    email: str
    password: str = Field(min_length=8, max_length=72)
    name: str = Field(default="", max_length=80)


class LoginRequest(BaseModel):
    email: str
    password: str = Field(max_length=72)


class AuthResponse(BaseModel):
    token: str
    user: dict


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/auth", tags=["auth"])
_db: Database | None = None


def configure(db: Database) -> None:
    global _db
    _db = db


def get_db() -> Database:
    if _db is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    return _db


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _load_user(db: Database, user_id: str) -> Optional[CurrentUser]:
    row = db.fetchone("SELECT id, email, name FROM users WHERE id = %s", (user_id,))
    if not row:
        return None
    return CurrentUser(id=row[0], email=row[1], name=row[2] or "")


def current_user(request: Request) -> CurrentUser:
    """FastAPI dependency: resolve the bearer token to a user or 401."""
    header = request.headers.get("authorization", "")
    token = header[7:].strip() if header.lower().startswith("bearer ") else ""
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sign in required")
    user_id = decode_token(token)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired, sign in again")
    user = _load_user(get_db(), user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Account no longer exists")
    return user


@router.post("/signup", response_model=AuthResponse)
def signup(req: SignupRequest):
    db = get_db()
    email = _normalize_email(req.email)
    if not EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="Enter a valid email address")
    if db.fetchone("SELECT 1 FROM users WHERE email = %s", (email,)):
        raise HTTPException(status_code=409, detail="An account with this email already exists")

    user_id = str(uuid.uuid4())
    name = req.name.strip() or email.split("@")[0]
    db.execute(
        "INSERT INTO users (id, email, name, password_hash) VALUES (%s, %s, %s, %s)",
        (user_id, email, name, hash_password(req.password)),
    )
    user = CurrentUser(id=user_id, email=email, name=name)
    return AuthResponse(token=create_token(user_id), user=user.public())


@router.post("/login", response_model=AuthResponse)
def login(req: LoginRequest):
    db = get_db()
    email = _normalize_email(req.email)
    row = db.fetchone("SELECT id, email, name, password_hash FROM users WHERE email = %s", (email,))
    if not row or not verify_password(req.password, row[3]):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    user = CurrentUser(id=row[0], email=row[1], name=row[2] or "")
    return AuthResponse(token=create_token(user.id), user=user.public())


@router.get("/me")
def me(user: CurrentUser = Depends(current_user)):
    return user.public()
