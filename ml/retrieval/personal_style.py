"""Şəxsi stil balı — istifadəçinin referens şəkilləri ilə şəkil↔şəkil oxşarlığı.

`style_scorer.py` şəkli MƏTN etiketləri ilə tutuşdurur ("bu nə qədər formal-dır").
Burada isə şəkil istifadəçinin ÖZ referens şəkilləri ilə tutuşdurulur ("bu nə
qədər MƏNİM zövqümdür"). İki fərqli sual, iki fərqli bal.

⚠️ MODALITY GAP — bu iki balı BİRBAŞA müqayisə etmək və ya toplamaq OLMAZ:

    style_scorer   : şəkil ↔ MƏTN  cosine, praktikada ~0.08-0.29
    personal_style : şəkil ↔ ŞƏKİL cosine, praktikada ~0.5-0.9

CLIP-də şəkil və mətn embedding-ləri ortaq fəzanın AYRI-AYRI konuslarında
yerləşir (məşhur "modality gap"). Ona görə `PERSONAL_SIM_THRESHOLD` (0.70)
`STYLE_TEXT_THRESHOLD`-dan (0.25) tamamilə fərqli diapazondadır — bunlar
müqayisə edilə bilən ədədlər DEYİL. Birləşdirmək lazım gələrsə əvvəlcə hər
biri öz diapazonunda normalize olunmalıdır (bax `normalize_for_fusion`).

AQREQASİYA (`config.PERSONAL_AGG`):
    "max"       — DEFAULT. Ən yaxın bir referansla oxşarlıq.
    "mean_top2" — Ən yaxın iki referansın ortası (bir təsadüfi uyğunluğa
                  qarşı bir az daha davamlı).
    Sadə ORTALAMA qəsdən yoxdur: zövq çoxmodallıdır. Eyni adam həm idman,
    həm klassik geyinə bilər — bütün referanslar üzrə ortalama HƏR İKİ
    klasteri cəzalandırır və heç birinə bənzəməyən "orta" şeyi mükafatlandırır.

SÜRƏT: referanslar HƏMİŞƏ tək SELECT ilə çəkilir və [R, 512] matris kimi
saxlanılır. Ballandırma tam matris hasilidir — Python döngüsü yoxdur.

İstifadə:
    python -m ml.retrieval.personal_style --add-refs u1 data/images/item_0001.jpg ...
    python -m ml.retrieval.personal_style --score u1 --top 10
    python -m ml.retrieval.personal_style --list u1
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

if __package__ in (None, ""):  # birbaşa `python personal_style.py` işlədiləndə
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.retrieval import config
from ml.retrieval.embedder import FashionCLIPEmbedder
from ml.retrieval.store.base import validate_vectors


def _as_array(value) -> np.ndarray:
    """pgvector sütununu [D] float32 massivə çevirir.

    `register_vector` pgvector versiyasından asılı olaraq ya numpy massivi,
    ya da `pgvector.Vector` obyekti qaytarır — ikisini də qəbul edirik.
    """
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    return np.asarray(value, dtype=np.float32)


def make_ref_id(user_id: str, image_path: str | Path) -> str:
    """`user_id` + şəkil yolundan deterministik id.

    Eyni şəkli iki dəfə əlavə etmək təkrar sətir yaratmır — upsert olur.
    """
    digest = hashlib.sha256(f"{user_id}\x1f{Path(image_path).as_posix()}".encode("utf-8"))
    return f"ref_{digest.hexdigest()[:16]}"


class PersonalStyle:
    """`user_style_refs` cədvəli üzərində referans idarəsi və ballandırma.

    pgvector backend TƏLƏB OLUNUR — referanslar DB-də yaşayır. numpy backend
    üçün ekvivalent yoxdur (bu, per-user datadır, paylaşılan artefakt deyil).
    """

    def __init__(
        self,
        db_url: str = config.DB_URL,
        table: str = config.STYLE_REFS_TABLE,
        embedder: FashionCLIPEmbedder | None = None,
        ensure_schema: bool = True,
    ) -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        if table != config.STYLE_REFS_TABLE:  # SQL string interpolyasiyası üçün ağ siyahı
            raise ValueError(f"İcazəsiz cədvəl adı: {table!r}")
        self.table = table
        self._embedder = embedder
        self.conn = psycopg.connect(db_url, autocommit=True)
        if ensure_schema:
            self._ensure_schema()
        register_vector(self.conn)

    @property
    def embedder(self) -> FashionCLIPEmbedder:
        """Model yalnız şəkil embed etmək lazım olanda yüklənir."""
        if self._embedder is None:
            self._embedder = FashionCLIPEmbedder()
        return self._embedder

    # --- sxem -------------------------------------------------------------
    def _ensure_schema(self) -> None:
        """init.sql ilə eyni sxem.

        `init.sql` yalnız BOŞ volume-da bir dəfə işləyir — artıq data olan
        DB-yə cədvəl bu yolla gəlir. Ona görə ikisi sinxron saxlanmalıdır.
        """
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    ref_id      TEXT PRIMARY KEY,
                    user_id     TEXT NOT NULL,
                    image_path  TEXT,
                    embedding   VECTOR({config.EMB_DIM}),
                    model_ver   TEXT DEFAULT '{config.MODEL_VER}',
                    created_at  TIMESTAMPTZ DEFAULT now()
                )
            """)
            # Adi B-tree — vektor indeksi qəsdən yoxdur (bax modul doc-u).
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table}_user_id_idx "
                f"ON {self.table} (user_id)"
            )

    # --- Task 3: referanslar ---------------------------------------------
    def add_style_refs(
        self,
        user_id: str,
        paths: Sequence[str | Path],
        model_ver: str = config.MODEL_VER,
    ) -> int:
        """Referens şəkilləri embed edib saxlayır. Yazılan sətir sayını qaytarır.

        Bütün şəkillər BİR `embed_images` çağırışı ilə (batch=8) və bütün
        sətirlər BİR `executemany` ilə yazılır.
        """
        paths = [Path(p) for p in paths]
        if not paths:
            return 0
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Şəkil tapılmadı: {[str(p) for p in missing]}")

        vectors = validate_vectors(self.embedder.embed_images(paths))
        rows = [
            (make_ref_id(user_id, path), user_id, str(path), vector, model_ver)
            for path, vector in zip(paths, vectors)
        ]
        with self.conn.cursor() as cur:
            cur.executemany(
                f"""
                INSERT INTO {self.table} (ref_id, user_id, image_path, embedding, model_ver)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (ref_id) DO UPDATE SET
                    image_path = EXCLUDED.image_path,
                    embedding  = EXCLUDED.embedding,
                    model_ver  = EXCLUDED.model_ver
                """,
                rows,
            )
        return len(rows)

    def get_refs(
        self, user_id: str, model_ver: str | None = config.MODEL_VER
    ) -> tuple[list[str], np.ndarray]:
        """İstifadəçinin referansları -> (ref_id siyahısı, [R, 512] matris).

        TƏK SELECT — N+1 sorğu problemi burada başlayır və burada bitir.
        `model_ver` verilibsə yalnız uyğun model versiyası çəkilir: köhnə
        modellə hesablanmış vektorla oxşarlıq ölçmək səssiz xətadır.
        """
        clause = "WHERE user_id = %s"
        params: list = [user_id]
        if model_ver is not None:
            clause += " AND model_ver = %s"
            params.append(model_ver)

        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT ref_id, embedding FROM {self.table} {clause} ORDER BY ref_id",
                params,
            )
            rows = cur.fetchall()

        if not rows:
            return [], np.zeros((0, config.EMB_DIM), dtype=np.float32)
        ref_ids = [row[0] for row in rows]
        vectors = np.stack([_as_array(row[1]) for row in rows])
        return ref_ids, validate_vectors(vectors)

    def list_refs(self, user_id: str) -> list[dict]:
        """Referansların metadata-sı (vektorsuz) — CLI/təftiş üçün."""
        with self.conn.cursor() as cur:
            cur.execute(
                f"""SELECT ref_id, image_path, model_ver, created_at
                    FROM {self.table} WHERE user_id = %s ORDER BY created_at, ref_id""",
                (user_id,),
            )
            return [
                {"ref_id": r[0], "image_path": r[1], "model_ver": r[2], "created_at": r[3]}
                for r in cur.fetchall()
            ]

    def delete_refs(self, user_id: str) -> int:
        with self.conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.table} WHERE user_id = %s", (user_id,))
            return cur.rowcount

    def count(self, user_id: str | None = None) -> int:
        with self.conn.cursor() as cur:
            if user_id is None:
                cur.execute(f"SELECT count(*) FROM {self.table}")
            else:
                cur.execute(f"SELECT count(*) FROM {self.table} WHERE user_id = %s", (user_id,))
            return int(cur.fetchone()[0])

    # --- Task 4: şəxsi bal ------------------------------------------------
    def personal_score(
        self,
        item_embs: np.ndarray,
        user_id: str,
        agg: str | None = None,
    ) -> np.ndarray:
        """[N, 512] əşya embedding-i -> [N] şəxsi stil balı.

        Referansı olmayan istifadəçi üçün [N] sıfır massivi qaytarılır
        (istisna yox: yeni istifadəçi normal haldır, sadəcə şəxsi siqnal yoxdur).
        """
        _, refs = self.get_refs(user_id)
        return aggregate_similarity(item_embs, refs, agg)

    def close(self) -> None:
        if getattr(self, "conn", None) is not None and not self.conn.closed:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --- saf funksiyalar (DB-siz test oluna bilir) ----------------------------
def aggregate_similarity(
    item_embs: np.ndarray,
    refs: np.ndarray,
    agg: str | None = None,
) -> np.ndarray:
    """[N, D] × [R, D] -> [N] bal. Tam matris hasili, Python döngüsü yoxdur."""
    agg = agg or config.PERSONAL_AGG
    if agg not in config.PERSONAL_AGG_CHOICES:
        raise ValueError(f"Naməlum PERSONAL_AGG: {agg!r} (mövcud: {config.PERSONAL_AGG_CHOICES})")

    item_embs = np.asarray(item_embs, dtype=np.float32)
    if item_embs.ndim == 1:
        item_embs = item_embs.reshape(1, -1)
    if len(refs) == 0:
        return np.zeros(len(item_embs), dtype=np.float32)

    similarity = item_embs @ np.asarray(refs, dtype=np.float32).T  # [N, R]

    if agg == "max":
        return similarity.max(axis=1).astype(np.float32)

    # mean_top2 — referans tək olanda top-2 yoxdur, max-a bərabər olur.
    k = min(2, similarity.shape[1])
    top_k = np.partition(similarity, -k, axis=1)[:, -k:]  # sıralamadan ucuz
    return top_k.mean(axis=1).astype(np.float32)


def normalize_for_fusion(scores: np.ndarray, low: float, high: float) -> np.ndarray:
    """Balı öz diapazonundan [0, 1]-ə gətirir — YALNIZ birləşdirmə üçün.

    Stil balı (şəkil↔mətn) və şəxsi bal (şəkil↔şəkil) fərqli diapazonlarda
    yaşayır. Onları toplamaq lazım gələrsə hər biri əvvəlcə ÖZ diapazonu ilə
    normalize olunmalıdır.

    `low`/`high`-ı GÖZDƏN yazma — MÜŞAHİDƏ OLUNAN paylanmadan götür:

        lo_s, hi_s = np.percentile(style_cosine, [5, 95])
        lo_p, hi_p = np.percentile(personal,     [5, 95])
        combined = 0.5 * normalize_for_fusion(style_cosine, lo_s, hi_s) \\
                 + 0.5 * normalize_for_fusion(personal,     lo_p, hi_p)

    50 item-lik dəstədə ölçülmüş töhfə balansı (50% ədalətli olardı):
        xam toplama                       -> şəxsi bal 70%  (stil əzilir)
        sabit diapazon (0.5-0.9 fərz edilib) -> şəxsi bal 19%  (şəxsi əzilir)
        data-dan p5-p95                   -> şəxsi bal 40%  ✓

    Yəni diapazonu səhv seçmək problemi HƏLL ETMİR, sadəcə istiqamətini
    dəyişir. `config.PERSONAL_SIM_THRESHOLD` və `STYLE_TEXT_THRESHOLD`
    hədd (qərar) üçündür — birləşdirmə miqyası üçün yox.
    """
    if high <= low:
        raise ValueError(f"high ({high}) low-dan ({low}) böyük olmalıdır")
    return np.clip((np.asarray(scores, dtype=np.float32) - low) / (high - low), 0.0, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Şəxsi stil balı (user_style_refs)")
    parser.add_argument("--add-refs", nargs="+", metavar=("USER_ID", "PATH"),
                        help="referens şəkilləri əlavə et: USER_ID path1 path2 ...")
    parser.add_argument("--score", metavar="USER_ID", help="qarderobu bu istifadəçi üçün ballandır")
    parser.add_argument("--list", metavar="USER_ID", dest="list_user")
    parser.add_argument("--delete", metavar="USER_ID")
    parser.add_argument("--agg", default=None, choices=config.PERSONAL_AGG_CHOICES)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--info", action="store_true", help="cədvəl və indeks vəziyyəti")
    args = parser.parse_args()

    with PersonalStyle() as personal:
        if args.info:
            with personal.conn.cursor() as cur:
                cur.execute(
                    """SELECT indexname, indexdef FROM pg_indexes WHERE tablename = %s""",
                    (personal.table,),
                )
                indexes = cur.fetchall()
            print(f"cədvəl : {personal.table}")
            print(f"sətir  : {personal.count()}")
            print("indekslər:")
            for name, definition in indexes:
                kind = ("ANN!" if "ivfflat" in definition.lower() or "hnsw" in definition.lower()
                        else "b-tree")
                print(f"  [{kind}] {name}")
            return 0

        if args.add_refs:
            user_id, paths = args.add_refs[0], args.add_refs[1:]
            if not paths:
                parser.error("ən azı bir şəkil yolu ver")
            written = personal.add_style_refs(user_id, paths)
            print(f"{written} referans yazıldı -> user_id={user_id} "
                  f"(cəmi {personal.count(user_id)})")
            return 0

        if args.list_user:
            refs = personal.list_refs(args.list_user)
            print(f"user_id={args.list_user}: {len(refs)} referans")
            for ref in refs:
                print(f"  {ref['ref_id']}  {ref['model_ver']:<16} {ref['image_path']}")
            return 0

        if args.delete:
            print(f"{personal.delete_refs(args.delete)} referans silindi")
            return 0

        if args.score:
            from ml.retrieval.style_scorer import load_wardrobe

            ids, embeddings = load_wardrobe()
            ref_ids, refs = personal.get_refs(args.score)
            if not len(refs):
                print(f"user_id={args.score} üçün referans yoxdur -> --add-refs")
                return 1
            agg = args.agg or config.PERSONAL_AGG
            scores = aggregate_similarity(embeddings, refs, agg)

            print(f"istifadəçi : {args.score}")
            print(f"referans   : {len(refs)} ədəd, matris {refs.shape}")
            print(f"aqreqasiya : {agg}")
            print(f"hədd       : PERSONAL_SIM_THRESHOLD = {config.PERSONAL_SIM_THRESHOLD} "
                  f"(şəkil↔ŞƏKİL diapazonu — STYLE_TEXT_THRESHOLD ilə müqayisə etmə!)")
            print(f"bal        : min={scores.min():.4f} max={scores.max():.4f} "
                  f"orta={scores.mean():.4f}")
            above = int((scores >= config.PERSONAL_SIM_THRESHOLD).sum())
            print(f"həddi keçən: {above}/{len(scores)}\n")

            order = np.argsort(-scores)[: args.top]
            print(f"=== top-{args.top} ===")
            for rank, index in enumerate(order, start=1):
                mark = "✓" if scores[index] >= config.PERSONAL_SIM_THRESHOLD else " "
                print(f"  {rank:>2}. {mark} {ids[index]:<14} {scores[index]:.4f}")

            if args.agg is None:  # hər iki aqreqasiyanı müqayisə et
                other = "mean_top2" if agg == "max" else "max"
                other_scores = aggregate_similarity(embeddings, refs, other)
                changed = int((np.argsort(-scores)[:5] != np.argsort(-other_scores)[:5]).sum())
                print(f"\n{other:<10}: orta={other_scores.mean():.4f} "
                      f"(fərq {other_scores.mean() - scores.mean():+.4f}), "
                      f"top-5-də dəyişən mövqe: {changed}")
            return 0

    parser.error("bir əməliyyat seç: --add-refs | --score | --list | --delete | --info")


if __name__ == "__main__":
    raise SystemExit(main())
