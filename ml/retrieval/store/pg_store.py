"""pgvector backend — PostgreSQL + `vector` uzantısı, exact cosine search.

Sxem `init.sql`-də: item_embeddings(item_id, image_path, category, color,
embedding VECTOR(512), model_ver, source, created_at).

ANN index QURULMUR (config.BUILD_ANN_INDEX = False) — exact `<=>` kifayətdir
və numpy backend-i ilə eyni nəticəni zəmanətləyir.

pgvector `<=>` cosine DISTANCE qaytarır: distance = 1 - cosine_similarity.
Biz hər yerdə similarity ilə işləyirik, ona görə `1 - (a <=> b)` yazırıq.

İstifadə:
    python -m ml.retrieval.store.pg_store --info
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ml.retrieval import config
from ml.retrieval.store.base import (
    SearchResult,
    VectorStore,
    as_query_vector,
    validate_vectors,
)

# meta dict-in hansı açarları hansı sütuna düşür
META_COLUMNS = ("image_path", "category", "color", "pattern", "model_ver", "source")
# WHERE filtrində icazə verilən sütunlar (SQL injection-a qarşı ağ siyahı)
FILTERABLE_COLUMNS = ("category", "color", "pattern", "source", "model_ver")


class PgStore(VectorStore):
    def __init__(
        self,
        db_url: str = config.DB_URL,
        table: str = config.EMB_TABLE,
        ensure_schema: bool = True,
    ) -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        if table not in (config.EMB_TABLE,):
            raise ValueError(f"İcazəsiz cədvəl adı: {table!r}")
        self.db_url = db_url
        self.table = table
        self.conn = psycopg.connect(db_url, autocommit=True)
        if ensure_schema:
            self._ensure_schema()
        register_vector(self.conn)

    # --- sxem -------------------------------------------------------------
    def _ensure_schema(self) -> None:
        """`vector` uzantısı və cədvəl yoxdursa yaradır (init.sql ilə eyni sxem)."""
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    item_id     TEXT PRIMARY KEY,
                    image_path  TEXT,
                    category    TEXT,
                    color       TEXT,
                    pattern     TEXT,
                    gender      TEXT,
                    embedding   VECTOR({config.EMB_DIM}),
                    model_ver   TEXT DEFAULT '{config.MODEL_VER}',
                    source      TEXT,
                    created_at  TIMESTAMPTZ DEFAULT now()
                )
            """)
        # config.BUILD_ANN_INDEX qəsdən False — exact search istəyirik.
        if config.BUILD_ANN_INDEX:
            raise NotImplementedError(
                "ANN index bu layihədə istifadə olunmur (exact search tələbi)."
            )

    # --- VectorStore ------------------------------------------------------
    def add(
        self,
        ids: Sequence[str],
        vecs: np.ndarray,
        meta: Sequence[dict[str, Any]] | None = None,
    ) -> int:
        ids = [str(i) for i in ids]
        vecs = validate_vectors(vecs)
        if len(ids) != len(vecs):
            raise ValueError(f"{len(ids)} id, {len(vecs)} vektor — sayları uyğun deyil")
        meta = list(meta) if meta is not None else [{} for _ in ids]
        if len(meta) != len(ids):
            raise ValueError(f"{len(ids)} id, {len(meta)} meta — sayları uyğun deyil")

        rows = [
            (
                item_id,
                item_meta.get("image_path"),
                item_meta.get("category"),
                item_meta.get("color"),
                item_meta.get("pattern"),
                item_meta.get("gender"),
                vec,
                item_meta.get("model_ver") or config.MODEL_VER,
                item_meta.get("source"),
            )
            for item_id, vec, item_meta in zip(ids, vecs, meta)
        ]
        with self.conn.cursor() as cur:
            cur.executemany(
                f"""
                INSERT INTO {self.table}
                    (item_id, image_path, category, color, pattern, gender, embedding, model_ver, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (item_id) DO UPDATE SET
                    image_path = EXCLUDED.image_path,
                    category   = EXCLUDED.category,
                    color      = EXCLUDED.color,
                    pattern    = EXCLUDED.pattern,
                    embedding  = EXCLUDED.embedding,
                    model_ver  = EXCLUDED.model_ver,
                    source     = EXCLUDED.source
                """,
                rows,
            )
        return len(rows)

    def search(
        self,
        vec: np.ndarray,
        k: int = config.DEFAULT_TOP_K,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        query = as_query_vector(vec)
        clauses, params = [], [query]
        for key, wanted in (where or {}).items():
            if key not in FILTERABLE_COLUMNS:
                raise ValueError(
                    f"Filtr sütunu dəstəklənmir: {key!r} (icazəli: {FILTERABLE_COLUMNS})"
                )
            if isinstance(wanted, (list, tuple, set)):
                clauses.append(f"{key} = ANY(%s)")
                params.append(list(wanted))
            else:
                clauses.append(f"{key} = %s")
                params.append(wanted)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([query, k])

        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT item_id,
                       1 - (embedding <=> %s) AS similarity,
                       image_path, category, color, pattern, model_ver, source
                FROM {self.table}
                {where_sql}
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                params,
            )
            rows = cur.fetchall()

        return [
            SearchResult(
                item_id=row[0],
                score=float(row[1]),
                meta={
                    "image_path": row[2],
                    "category": row[3],
                    "color": row[4],
                    "pattern": row[5],
                    "model_ver": row[6],
                    "source": row[7],
                },
            )
            for row in rows
        ]

    def count(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {self.table}")
            return int(cur.fetchone()[0])

    def get(self, item_id: str) -> SearchResult | None:
        with self.conn.cursor() as cur:
            cur.execute(
                f"""SELECT item_id, image_path, category, color, pattern, model_ver, source
                    FROM {self.table} WHERE item_id = %s""",
                (item_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return SearchResult(
            item_id=row[0],
            score=1.0,
            meta={
                "image_path": row[1],
                "category": row[2],
                "color": row[3],
                "pattern": row[4],
                "model_ver": row[5],
                "source": row[6],
            },
        )

    def delete(self, item_id: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.table} WHERE item_id = %s", (item_id,))
        self.conn.commit()

    # --- əlavə ------------------------------------------------------------
    def distinct_values(self, field: str) -> list[str]:
        if field not in FILTERABLE_COLUMNS:
            raise ValueError(
                f"Sütun dəstəklənmir: {field!r} (icazəli: {FILTERABLE_COLUMNS})"
            )
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT DISTINCT {field} FROM {self.table} "
                f"WHERE {field} IS NOT NULL ORDER BY {field}"
            )
            return [r[0] for r in cur.fetchall()]

    def clear(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.table}")
            return cur.rowcount

    def indexes(self) -> list[str]:
        """Cədvəldəki bütün indeks adları."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT indexname FROM pg_indexes WHERE tablename = %s", (self.table,))
            return [r[0] for r in cur.fetchall()]

    def ann_indexes(self) -> list[str]:
        """Yalnız ANN (ivfflat/hnsw) indekslər — olmamalıdır.

        Ada görə yox, indeks tərifinə (`USING ivfflat|hnsw`) görə yoxlanılır:
        PK indeksinin adı `item_embeddings_pkey`-dir və "embedding" alt-sətrini
        ehtiva edir, ona görə ad üzrə yoxlama yanlış nəticə verir.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT indexname FROM pg_indexes
                   WHERE tablename = %s
                     AND (indexdef ILIKE %s OR indexdef ILIKE %s)""",
                (self.table, "%USING ivfflat%", "%USING hnsw%"),
            )
            return [r[0] for r in cur.fetchall()]

    def close(self) -> None:
        if getattr(self, "conn", None) is not None and not self.conn.closed:
            self.conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="pgvector store məlumatı")
    parser.add_argument("--db-url", default=config.DB_URL)
    parser.add_argument("--head", type=int, default=5)
    parser.add_argument("--clear", action="store_true", help="cədvəli boşalt")
    args = parser.parse_args()

    with PgStore(db_url=args.db_url) as store:
        if args.clear:
            print(f"silindi: {store.clear()} sətir")
        with store.conn.cursor() as cur:
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            row = cur.fetchone()
        print(f"db_url    : {args.db_url}")
        print(f"pgvector  : {row[0] if row else '(quraşdırılmayıb)'}")
        print(f"cədvəl    : {store.table}")
        print(f"count     : {store.count()}")
        print(f"indekslər : {store.indexes()}")
        print(f"ANN index : {store.ann_indexes() or '(yoxdur — exact search)'}")
        with store.conn.cursor() as cur:
            cur.execute(
                f"SELECT item_id, category, source FROM {store.table} "
                f"ORDER BY item_id LIMIT %s", (args.head,)
            )
            for r in cur.fetchall():
                print(f"  {r[0]}  [{r[1]}]  {r[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
