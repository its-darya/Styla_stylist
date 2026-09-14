-- Styla — database schema (PostgreSQL + pgvector)
--
-- Runs automatically on a fresh Docker volume (docker-compose.yml). The
-- backend also applies the same statements on every start
-- (backend/db.py: ensure_schema), so an existing database — including the
-- shared Neon instance — is upgraded in place.

CREATE EXTENSION IF NOT EXISTS vector;

-- Accounts -----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    email         TEXT NOT NULL UNIQUE,
    name          TEXT,
    password_hash TEXT NOT NULL,          -- bcrypt
    created_at    TIMESTAMPTZ DEFAULT now()
);

-- Wardrobe (one row per garment, owned by a user) ---------------------------
CREATE TABLE IF NOT EXISTS item_embeddings (
    item_id     TEXT PRIMARY KEY,
    image_path  TEXT,
    category    TEXT,
    color       TEXT,
    pattern     TEXT,
    gender      TEXT,
    user_id     TEXT,
    embedding   VECTOR(512),
    model_ver   TEXT DEFAULT 'fashionclip-v1',
    source      TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);
ALTER TABLE item_embeddings ADD COLUMN IF NOT EXISTS gender  TEXT;
ALTER TABLE item_embeddings ADD COLUMN IF NOT EXISTS user_id TEXT;
CREATE INDEX IF NOT EXISTS item_embeddings_user_id_idx ON item_embeddings (user_id);

-- Qeyd: ANN index (ivfflat/hnsw) QƏSDƏN qurulmur.
-- MVP ölçüsündə exact search (`<=>`) kifayətdir və dəqiq nəticə verir.

-- İstifadəçinin stil referens şəkilləri (C: personal_style.py).
-- item_embeddings QARDEROBDUR (sahib olduğu əşyalar); bu isə ZÖVQ NÜMUNƏSİDİR
-- ("belə geyinmək istəyirəm") — ayrı həyat dövrü, ayrı cədvəl.
CREATE TABLE IF NOT EXISTS user_style_refs (
    ref_id      TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    image_path  TEXT,
    embedding   VECTOR(512),
    model_ver   TEXT DEFAULT 'fashionclip-v1',
    created_at  TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS user_style_refs_user_id_idx ON user_style_refs (user_id);

-- Saved looks --------------------------------------------------------------
CREATE TABLE IF NOT EXISTS saved_outfits (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    style       TEXT NOT NULL,
    items       JSONB NOT NULL,
    score       REAL,
    created_at  TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS saved_outfits_user_id_idx ON saved_outfits (user_id);
