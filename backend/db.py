"""Application database layer (PostgreSQL + pgvector).

One database serves the whole app. `DATABASE_URL` (from `.env`) selects it:
the team's shared Neon Postgres in production, or the local Docker pgvector
from `docker-compose.yml` when developing offline. Both need the `vector`
extension; nothing else is required.

This module owns:

- `Database`          — a single long-lived psycopg connection that
                        transparently reconnects when the server (Neon's
                        pooler in particular) drops an idle connection.
- `ensure_schema()`   — idempotent migrations for the application tables
                        (users, saved_outfits) and the per-user columns on
                        the ML tables (item_embeddings.user_id).
- `ensure_demo_user()`— creates the built-in demo account and adopts any
                        wardrobe rows that pre-date multi-user support, so
                        the seeded wardrobe stays usable after the upgrade.

The ML retrieval modules (`ml/retrieval/store/pg_store.py`,
`ml/retrieval/personal_style.py`) keep their own connections; they read the
same `DATABASE_URL` through `ml.retrieval.config.DB_URL`.
"""
from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import psycopg

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_DB_URL = "postgresql://styla_user:styla_password@localhost:5440/styla_wardrobe"
DB_URL = os.getenv("DATABASE_URL") or os.getenv("STYLA_DB_URL") or DEFAULT_DB_URL

DEMO_EMAIL = os.getenv("STYLA_DEMO_EMAIL", "demo@styla.app")
DEMO_PASSWORD = os.getenv("STYLA_DEMO_PASSWORD", "demo1234")
DEMO_NAME = "Demo User"

# user_id values written by the pre-auth frontend/backend for style refs.
LEGACY_USER_IDS = ("default_user", "user123")

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    email         TEXT NOT NULL UNIQUE,
    name          TEXT,
    password_hash TEXT NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT now()
);

-- Wardrobe embeddings live in item_embeddings (see init.sql). Each row now
-- belongs to exactly one user.
CREATE TABLE IF NOT EXISTS item_embeddings (
    item_id     TEXT PRIMARY KEY,
    image_path  TEXT,
    category    TEXT,
    color       TEXT,
    pattern     TEXT,
    gender      TEXT,
    embedding   VECTOR(512),
    model_ver   TEXT DEFAULT 'fashionclip-v1',
    source      TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);
ALTER TABLE item_embeddings ADD COLUMN IF NOT EXISTS gender  TEXT;
ALTER TABLE item_embeddings ADD COLUMN IF NOT EXISTS user_id TEXT;
CREATE INDEX IF NOT EXISTS item_embeddings_user_id_idx ON item_embeddings (user_id);

CREATE TABLE IF NOT EXISTS user_style_refs (
    ref_id      TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    image_path  TEXT,
    embedding   VECTOR(512),
    model_ver   TEXT DEFAULT 'fashionclip-v1',
    created_at  TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS user_style_refs_user_id_idx ON user_style_refs (user_id);

-- Looks the user saved from the generator. `items` is the outfit payload as
-- shown in the UI (id, imageUrl, category, color, pattern, gender).
CREATE TABLE IF NOT EXISTS saved_outfits (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    style       TEXT NOT NULL,
    items       JSONB NOT NULL,
    score       REAL,
    created_at  TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS saved_outfits_user_id_idx ON saved_outfits (user_id);
"""


def _prepare(conn: psycopg.Connection) -> None:
    """Per-connection setup: pgvector adapters, if the extension is present."""
    try:
        from pgvector.psycopg import register_vector

        register_vector(conn)
    except Exception:  # pgvector adapter is optional for plain queries
        pass


class Database:
    """A pooled set of autocommit connections.

    A pool rather than one shared connection, for two reasons. FastAPI runs
    sync endpoints in a thread pool, and a psycopg connection cannot be used
    by two threads at once — sharing one produces sporadic "another operation
    in progress" failures under even light concurrency. And Neon drops idle
    connections after a few minutes, which the pool notices and replaces
    before handing one out, so a query never lands on a dead socket.
    """

    def __init__(self, db_url: str = DB_URL, min_size: int = 1, max_size: int = 8) -> None:
        from psycopg_pool import ConnectionPool

        self.db_url = db_url
        self._pool = ConnectionPool(
            conninfo=db_url,
            min_size=min_size,
            max_size=max_size,
            kwargs={"autocommit": True, "connect_timeout": 20},
            configure=_prepare,
            # Hand out only connections that still answer, so a link Neon
            # closed while idle is replaced instead of raising mid-request.
            check=ConnectionPool.check_connection,
            open=True,
            timeout=30,
        )

    @contextmanager
    def connection(self) -> Iterator[psycopg.Connection]:
        """Borrow a connection for the duration of the block."""
        with self._pool.connection() as conn:
            yield conn

    @contextmanager
    def cursor(self) -> Iterator[psycopg.Cursor]:
        """Borrow a connection and yield a cursor on it.

        Note there is deliberately no retry here: a `with` body cannot be
        re-run from inside a context manager (the generator would yield
        twice, which raises "generator didn't stop after throw"). Recovery
        belongs to the pool's connection check instead.
        """
        with self._pool.connection() as conn, conn.cursor() as cur:
            yield cur

    def fetchall(self, sql: str, params: Any = None) -> list[tuple]:
        with self.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def fetchone(self, sql: str, params: Any = None) -> tuple | None:
        with self.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()

    def execute(self, sql: str, params: Any = None) -> int:
        with self.cursor() as cur:
            cur.execute(sql, params)
            return cur.rowcount

    def close(self) -> None:
        self._pool.close()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def ensure_schema(db: Database) -> None:
    """Create/upgrade all application tables. Safe to run on every start."""
    with db.cursor() as cur:
        cur.execute(SCHEMA_SQL)


def ensure_demo_user(db: Database) -> str:
    """Create the demo account (if missing) and adopt legacy rows.

    Rows written before accounts existed have `user_id IS NULL`
    (wardrobe) or one of `LEGACY_USER_IDS` (style refs). They are assigned
    to the demo user so the seeded wardrobe stays visible.

    Returns the demo user's id.
    """
    from backend.auth import hash_password  # local import: auth imports db

    row = db.fetchone("SELECT id FROM users WHERE email = %s", (DEMO_EMAIL,))
    if row:
        demo_id = row[0]
    else:
        demo_id = str(uuid.uuid4())
        db.execute(
            "INSERT INTO users (id, email, name, password_hash) VALUES (%s, %s, %s, %s)",
            (demo_id, DEMO_EMAIL, DEMO_NAME, hash_password(DEMO_PASSWORD)),
        )
        print(f"[db] created demo account {DEMO_EMAIL}")

    adopted = db.execute(
        "UPDATE item_embeddings SET user_id = %s WHERE user_id IS NULL", (demo_id,)
    )
    adopted += db.execute(
        "UPDATE user_style_refs SET user_id = %s WHERE user_id = ANY(%s)",
        (demo_id, list(LEGACY_USER_IDS)),
    )
    if adopted:
        print(f"[db] assigned {adopted} legacy rows to {DEMO_EMAIL}")
    return demo_id


def main() -> int:
    """`python -m backend.db` — apply migrations and print a short status."""
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
    db = Database(os.getenv("DATABASE_URL") or DB_URL)
    ensure_schema(db)
    demo_id = ensure_demo_user(db)
    host = db.db_url.split("@")[-1].split("/")[0]
    print(f"database : {host}")
    print(f"users    : {db.fetchone('SELECT count(*) FROM users')[0]}")
    print(f"wardrobe : {db.fetchone('SELECT count(*) FROM item_embeddings')[0]} items")
    print(f"looks    : {db.fetchone('SELECT count(*) FROM saved_outfits')[0]}")
    print(f"demo id  : {demo_id}")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
