-- Styla — verilənlər bazası sxemi
-- pgvector uzantısı və retrieval modulunun embedding cədvəli.

CREATE EXTENSION IF NOT EXISTS vector;

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

-- Adi B-tree — vektor indeksi (ivfflat/hnsw) QƏSDƏN yoxdur.
-- Sorğu həmişə `WHERE user_id = %s`; vektorlar TƏK SELECT ilə çəkilib
-- numpy-da matris kimi işlənir, ona görə ANN indeksinin faydası yoxdur.
CREATE INDEX IF NOT EXISTS user_style_refs_user_id_idx ON user_style_refs (user_id);
