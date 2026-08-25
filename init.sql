-- Styla — verilənlər bazası sxemi
-- pgvector uzantısı və retrieval modulunun embedding cədvəli.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS item_embeddings (
    item_id     TEXT PRIMARY KEY,
    image_path  TEXT,
    category    TEXT,
    color       TEXT,
    embedding   VECTOR(512),
    model_ver   TEXT DEFAULT 'fashionclip-v1',
    source      TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Qeyd: ANN index (ivfflat/hnsw) QƏSDƏN qurulmur.
-- MVP ölçüsündə exact search (`<=>`) kifayətdir və dəqiq nəticə verir.
