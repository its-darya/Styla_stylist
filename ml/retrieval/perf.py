"""Profiling köməkçiləri və outfit tövsiyəsi benchmark-ı.

Ölçülən ssenari (real istifadəçi sorğusu):
    istifadəçi bir referens şəkil verir -> qarderobundan top-5 outfit təklifi
    (bir üst + bir alt + bir ayaqqabı)

Dörd mərhələ AYRICA ölçülür — `time.perf_counter()` ilə:
    embed         : referens şəklin embedding-i (model, CPU)
    db            : qarderob + istifadəçi referansları
    scoring       : stil balı + şəxsi bal
    kombinatorika : slot təyini + outfit sıralaması

DİQQƏT — pgvector indeksi (ivfflat/hnsw) QURULMUR. Bizim miqyasda exact
search millisaniyələrlə ölçülür; ANN indeksi recall itkisi gətirər və heç bir
şey qazandırmazdı. Darboğaz başqa yerdədir — bu modul onu rəqəmlə göstərir.

İstifadə:
    python -m ml.retrieval.perf --baseline --user u_dress
    python -m ml.retrieval.perf --baseline --scale 2000
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

if __package__ in (None, ""):  # birbaşa `python perf.py` işlədiləndə
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.retrieval import config

BENCH_SOURCE = "perf_bench"  # sintetik sətirlərin işarəsi (təmizləmə üçün)


# --- profiling ------------------------------------------------------------
class Profile:
    """Adlandırılmış mərhələlərin müddətlərini toplayır.

    Eyni ad bir neçə dəfə ölçülə bilər (məs. N+1 sorğu) — hər çağırış ayrıca
    yazılır, ona görə həm CƏMİ, həm ÇAĞIRIŞ SAYI görünür. N+1 problemini
    məhz çağırış sayı ifşa edir.
    """

    def __init__(self, label: str = "") -> None:
        self.label = label
        self.durations: dict[str, list[float]] = {}
        self.counters: dict[str, int] = {}

    def record(self, name: str, seconds: float) -> None:
        self.durations.setdefault(name, []).append(seconds)

    def count(self, name: str, amount: int = 1) -> None:
        """Vaxtdan kənar sayğac (məs. SQL sorğu sayı, yoxlanan kombinasiya)."""
        self.counters[name] = self.counters.get(name, 0) + amount

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, time.perf_counter() - start)

    def total(self, name: str | None = None) -> float:
        if name is not None:
            return sum(self.durations.get(name, ()))
        return sum(sum(v) for v in self.durations.values())

    def calls(self, name: str) -> int:
        return len(self.durations.get(name, ()))

    def table(self, width: int = 18) -> str:
        total = self.total() or 1e-12
        header = (f"{'mərhələ':<{width}}{'san':>9}{'%':>7}{'çağırış':>9}"
                  f"{'ms/çağırış':>12}")
        lines = [header, "-" * len(header)]
        for name in sorted(self.durations, key=lambda n: -self.total(n)):
            seconds = self.total(name)
            calls = self.calls(name)
            lines.append(
                f"{name:<{width}}{seconds:>9.3f}{100 * seconds / total:>6.1f}%"
                f"{calls:>9}{1000 * seconds / calls:>12.3f}"
            )
        lines.append("-" * len(header))
        lines.append(f"{'CƏMİ':<{width}}{self.total():>9.3f}{100.0:>6.1f}%")
        if self.counters:
            lines.append("")
            for name, value in sorted(self.counters.items()):
                lines.append(f"  {name}: {value:,}")
        return "\n".join(lines)


@contextmanager
def timed(name: str = "") -> Iterator[list]:
    """Tək ölçmə — Profile lazım olmayan yerlər üçün.

        with timed("embed") as t:
            ...
        print(t[0])   # saniyə
    """
    holder: list = [0.0]
    start = time.perf_counter()
    try:
        yield holder
    finally:
        holder[0] = time.perf_counter() - start
        if name:
            print(f"[perf] {name}: {holder[0]:.3f} san")


# --- slot köməkçiləri -----------------------------------------------------
def coarse_to_slot() -> dict[str, str]:
    """`CATEGORIES` etiketi -> outfit slotu ("top" | "bottom" | "shoes")."""
    return {
        label: slot
        for slot, labels in config.OUTFIT_SLOTS.items()
        for label in labels
    }


def score_outfit(style_row: np.ndarray, personal_value: float) -> float:
    """Bir əşyanın outfit balına töhfəsi.

    Stil balı olaraq ən güclü stilin XAM cosine-i götürülür (item-lərarası
    müqayisə), şəxsi bal isə öz diapazonundadır. İkisi TOPLANMIR — hər biri
    ayrıca qaytarılıb yuxarıda normalize olunur (bax personal_style
    `normalize_for_fusion`). Burada sadəcə benchmark üçün cəm istifadə edilir.
    """
    return float(style_row.max()) + float(personal_value)


# --- SADƏLÖVH baseline (qəsdən pis) ---------------------------------------
def recommend_naive(
    user_id: str,
    reference_image: str | Path,
    profile: Profile,
    top_k: int = 5,
) -> list[tuple[float, tuple[str, str, str]]]:
    """Tipik "işləyir, amma yavaş" implementasiya — ÖLÇMƏ ÜÇÜN.

    Qəsdən edilmiş dörd səhv:
      1. Hər əşyanın embedding-i AYRICA SELECT ilə çəkilir (N+1 sorğu)
      2. Hər referansın embedding-i AYRICA SELECT ilə çəkilir (N+1 sorğu)
      3. Bütün ballandırma iç-içə Python döngüsü ilə (matris hasili yox)
      4. Kombinatorika: slotdakı BÜTÜN əşyalar üzrə tam Dekart hasili
    """
    import json

    import psycopg
    from pgvector.psycopg import register_vector

    from ml.retrieval.embedder import FashionCLIPEmbedder
    from ml.retrieval.matcher import CategoryClassifier
    from ml.retrieval.personal_style import _as_array
    from ml.retrieval.style_prompts import load_style_embeddings

    # --- 0. hazırlıq ------------------------------------------------------
    # Model yüklənməsi AYRICA ölçülür: bu, sorğu-başına deyil, proses-başına
    # xərcdir (uzun ömürlü servisdə bir dəfə olur). `embed` mərhələsinin
    # içində qalsa, ölçmə yanıldıcı olar — ilk versiyada məhz belə olmuşdu.
    embedder = FashionCLIPEmbedder()
    with profile.stage("setup_model_load"):
        embedder._ensure_loaded()
    with profile.stage("setup_style_embs"):
        style_embs = load_style_embeddings(embedder)

    # --- 1. embed (təmiz — model artıq yüklüdür) --------------------------
    with profile.stage("embed"):
        embedder.embed_images([reference_image])

    conn = psycopg.connect(config.DB_URL, autocommit=True)
    register_vector(conn)
    try:
        # --- 2. DB: N+1 sorğu --------------------------------------------
        with profile.stage("db_ids"):
            with conn.cursor() as cur:
                cur.execute(f"SELECT item_id FROM {config.EMB_TABLE} ORDER BY item_id")
                item_ids = [r[0] for r in cur.fetchall()]
            profile.count("sql_sorğu")

        vectors = []
        with profile.stage("db_wardrobe_n+1"):
            for item_id in item_ids:  # <-- N+1: hər əşya üçün bir sorğu
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT embedding FROM {config.EMB_TABLE} WHERE item_id = %s",
                        (item_id,),
                    )
                    vectors.append(_as_array(cur.fetchone()[0]))
                profile.count("sql_sorğu")
        wardrobe = np.stack(vectors)

        with profile.stage("db_refs_n+1"):
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT ref_id FROM {config.STYLE_REFS_TABLE} WHERE user_id = %s",
                    (user_id,),
                )
                ref_ids = [r[0] for r in cur.fetchall()]
            profile.count("sql_sorğu")
            refs = []
            for ref_id in ref_ids:  # <-- N+1 yenə
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT embedding FROM {config.STYLE_REFS_TABLE} WHERE ref_id = %s",
                        (ref_id,),
                    )
                    refs.append(_as_array(cur.fetchone()[0]))
                profile.count("sql_sorğu")
        refs = np.stack(refs) if refs else np.zeros((0, config.EMB_DIM), dtype=np.float32)
    finally:
        conn.close()

    # --- 3. scoring: Python döngüsü --------------------------------------
    style_matrix = style_embs.vectors
    with profile.stage("scoring_style_loop"):
        style_scores = np.zeros((len(wardrobe), len(style_matrix)), dtype=np.float32)
        for i in range(len(wardrobe)):          # <-- döngü, matris hasili yox
            for j in range(len(style_matrix)):
                style_scores[i, j] = float(np.dot(wardrobe[i], style_matrix[j]))

    with profile.stage("scoring_personal_loop"):
        personal_scores = np.zeros(len(wardrobe), dtype=np.float32)
        for i in range(len(wardrobe)):          # <-- döngü yenə
            best = 0.0
            for r in range(len(refs)):
                best = max(best, float(np.dot(wardrobe[i], refs[r])))
            personal_scores[i] = best

    # --- 4. slot təyini: hər əşya üçün ayrıca təsnifat --------------------
    # Kateqoriya promptlarının embed edilməsi də hazırlıqdır (18 mətn).
    with profile.stage("setup_category_prompts"):
        classifier = CategoryClassifier(embedder)
    slot_of = coarse_to_slot()
    with profile.stage("slot_təyini_loop"):
        slots: dict[str, list[int]] = {s: [] for s in config.OUTFIT_SLOTS}
        for i in range(len(wardrobe)):          # <-- əşya-əşya
            coarse, _ = classifier.classify_vector(wardrobe[i])
            slot = slot_of.get(coarse)
            if slot is not None:
                slots[slot].append(i)

    # --- 5. kombinatorika: TAM Dekart hasili ------------------------------
    with profile.stage("kombinatorika_tam"):
        combos = []
        for top, bottom, shoes in itertools.product(
            slots["top"], slots["bottom"], slots["shoes"]  # <-- top-K filtri YOX
        ):
            total = (
                score_outfit(style_scores[top], personal_scores[top])
                + score_outfit(style_scores[bottom], personal_scores[bottom])
                + score_outfit(style_scores[shoes], personal_scores[shoes])
            )
            combos.append((total, (item_ids[top], item_ids[bottom], item_ids[shoes])))
            profile.count("yoxlanan_kombinasiya")
        combos.sort(key=lambda row: -row[0])    # <-- hamısını sırala, heapq yox

    for slot, members in slots.items():
        profile.count(f"slot_{slot}", len(members))
    return combos[:top_k]


# --- OPTİMALLAŞDIRILMIŞ yol -----------------------------------------------
def recommend_fast(
    user_id: str,
    reference_image: str | Path,
    profile: Profile,
    top_k: int = 5,
    candidates: int = config.CANDIDATES_PER_CATEGORY,
) -> list[tuple[float, tuple[str, str, str]]]:
    """Eyni nəticə, dörd düzəlişlə.

      1. Qarderob TƏK SELECT ilə -> [N, 512] matris (N+1 sorğu yoxdur)
      2. Referanslar TƏK SELECT ilə -> [R, 512] matris
      3. Bütün ballandırma matris hasili (Python döngüsü yoxdur)
      4. Hər slotdan yalnız top-K namizəd -> K³ kombinasiya (Dekart hasili yox)

    Nəticə sadəlövh yolla EYNİ olmalıdır: outfit balı əşya ballarının CƏMİdir,
    ona görə ən yaxşı outfit-lər mütləq ən yaxşı əşyalardan qurulur —
    top-K kəsimi düzgün cavabı ata bilməz (K >= top_k şərti ilə).
    """
    import psycopg
    from pgvector.psycopg import register_vector

    from ml.retrieval.embedder import FashionCLIPEmbedder
    from ml.retrieval.matcher import CategoryClassifier
    from ml.retrieval.personal_style import _as_array
    from ml.retrieval.style_prompts import load_style_embeddings

    embedder = FashionCLIPEmbedder()
    with profile.stage("setup_model_load"):
        embedder._ensure_loaded()
    with profile.stage("setup_style_embs"):
        style_embs = load_style_embeddings(embedder)
    with profile.stage("setup_category_prompts"):
        classifier = CategoryClassifier(embedder)

    with profile.stage("embed"):
        embedder.embed_images([reference_image])

    conn = psycopg.connect(config.DB_URL, autocommit=True)
    register_vector(conn)
    try:
        # --- 1+2. DB: iki sorğu, vəssalam ---------------------------------
        with profile.stage("db_wardrobe_tək"):
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT item_id, embedding FROM {config.EMB_TABLE} ORDER BY item_id"
                )
                rows = cur.fetchall()
            profile.count("sql_sorğu")
            item_ids = [r[0] for r in rows]
            wardrobe = np.stack([_as_array(r[1]) for r in rows])

        with profile.stage("db_refs_tək"):
            with conn.cursor() as cur:
                cur.execute(
                    f"""SELECT embedding FROM {config.STYLE_REFS_TABLE}
                        WHERE user_id = %s AND model_ver = %s""",
                    (user_id, config.MODEL_VER),
                )
                ref_rows = cur.fetchall()
            profile.count("sql_sorğu")
            refs = (np.stack([_as_array(r[0]) for r in ref_rows]) if ref_rows
                    else np.zeros((0, config.EMB_DIM), dtype=np.float32))
    finally:
        conn.close()

    # --- 3. scoring: iki matris hasili ------------------------------------
    with profile.stage("scoring_matris"):
        style_scores = wardrobe @ style_embs.vectors.T                    # [N, 8]
        personal_scores = (
            (wardrobe @ refs.T).max(axis=1) if len(refs)
            else np.zeros(len(wardrobe), dtype=np.float32)
        )                                                                 # [N]
        item_scores = style_scores.max(axis=1) + personal_scores          # [N]

    # --- 4. slot təyini: bir matris hasili + argmax -----------------------
    with profile.stage("slot_təyini_matris"):
        category_scores = wardrobe @ classifier._prompt_vectors.T         # [N, 18]
        best_category = np.argmax(category_scores, axis=1)
        slot_of = coarse_to_slot()
        # kateqoriya indeksi -> slot indeksi (yoxdursa -1)
        slot_names = list(config.OUTFIT_SLOTS)
        category_to_slot = np.array([
            slot_names.index(slot_of[c]) if c in slot_of else -1
            for c in classifier.categories
        ])
        item_slot = category_to_slot[best_category]                       # [N]

    # --- 5. kombinatorika: slot başına top-K, sonra K³ ---------------------
    with profile.stage("kombinatorika_topk"):
        picks: dict[str, np.ndarray] = {}
        for position, name in enumerate(slot_names):
            members = np.flatnonzero(item_slot == position)
            if len(members) == 0:
                picks[name] = members
                continue
            scores = item_scores[members]
            k = min(candidates, len(members))
            # argpartition: tam sıralamadan ucuz (O(n) vs O(n log n))
            chosen = members[np.argpartition(-scores, k - 1)[:k]]
            picks[name] = chosen[np.argsort(-item_scores[chosen])]

        tops, bottoms, shoes = (picks[n] for n in ("top", "bottom", "shoes"))
        if not (len(tops) and len(bottoms) and len(shoes)):
            return []

        # Yayım (broadcasting) ilə [K, K, K] cəm — Python döngüsü yoxdur.
        totals = (
            item_scores[tops][:, None, None]
            + item_scores[bottoms][None, :, None]
            + item_scores[shoes][None, None, :]
        )
        profile.count("yoxlanan_kombinasiya", int(totals.size))

        flat = totals.ravel()
        k = min(top_k, flat.size)
        best_flat = np.argpartition(-flat, k - 1)[:k]
        best_flat = best_flat[np.argsort(-flat[best_flat])]
        a, b, c = np.unravel_index(best_flat, totals.shape)

    for name, members in picks.items():
        profile.count(f"slot_{name}_namizəd", len(members))
    for position, name in enumerate(slot_names):
        profile.count(f"slot_{name}", int((item_slot == position).sum()))

    return [
        (float(flat[i]), (item_ids[tops[x]], item_ids[bottoms[y]], item_ids[shoes[z]]))
        for i, x, y, z in zip(best_flat, a, b, c)
    ]


# --- sintetik miqyas ------------------------------------------------------
def seed_synthetic(count: int) -> int:
    """Benchmark üçün `count` sintetik sətir əlavə edir (source='perf_bench').

    Vektorlar təsadüfidir — perf semantikadan asılı deyil. `clear_synthetic()`
    ilə tam təmizlənir; benchmark bunu `finally`-də özü edir.
    """
    import psycopg
    from pgvector.psycopg import register_vector

    from ml.retrieval.embedder import l2_normalize

    rng = np.random.default_rng(0)
    vectors = l2_normalize(rng.standard_normal((count, config.EMB_DIM)).astype(np.float32))
    # Kateqoriyalar slotlar arasında bərabər paylanır ki, kombinatorika real olsun.
    labels = [l for labels in config.OUTFIT_SLOTS.values() for l in labels]

    conn = psycopg.connect(config.DB_URL, autocommit=True)
    register_vector(conn)
    try:
        with conn.cursor() as cur:
            cur.executemany(
                f"""INSERT INTO {config.EMB_TABLE}
                        (item_id, image_path, category, embedding, model_ver, source)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (item_id) DO NOTHING""",
                [
                    (f"bench_{i:06d}", None, labels[i % len(labels)],
                     vectors[i], config.MODEL_VER, BENCH_SOURCE)
                    for i in range(count)
                ],
            )
            return cur.rowcount
    finally:
        conn.close()


def clear_synthetic() -> int:
    """Bütün `source='perf_bench'` sətirlərini silir."""
    import psycopg

    conn = psycopg.connect(config.DB_URL, autocommit=True)
    try:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {config.EMB_TABLE} WHERE source = %s", (BENCH_SOURCE,))
            return cur.rowcount
    finally:
        conn.close()


def wardrobe_size() -> int:
    import psycopg

    conn = psycopg.connect(config.DB_URL, autocommit=True)
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {config.EMB_TABLE}")
            return int(cur.fetchone()[0])
    finally:
        conn.close()


def _request_seconds(profile: Profile) -> float:
    """`setup_*` mərhələləri çıxılmış vaxt — yəni SORĞU-başına xərc.

    Model yüklənməsi və prompt embed-ləri proses-başınadır; uzun ömürlü
    servisdə hər sorğuya düşmür, ona görə hədəflə müqayisədə sayılmır.
    """
    setup = sum(profile.total(n) for n in profile.durations if n.startswith("setup_"))
    return profile.total() - setup


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieval perf benchmark")
    parser.add_argument("--baseline", action="store_true", help="sadəlövh implementasiyanı ölç")
    parser.add_argument("--user", default="u_dress")
    parser.add_argument("--reference", default=None, help="referens şəkil (default: ilk şəkil)")
    parser.add_argument("--scale", type=int, default=0,
                        help="benchmark üçün bu qədər sintetik sətir əlavə et")
    parser.add_argument("--keep-synthetic", action="store_true",
                        help="sintetik sətirləri silmə (default: silinir)")
    parser.add_argument("--clear", action="store_true", help="yalnız sintetik sətirləri sil")
    parser.add_argument("--fast", action="store_true", help="optimallaşdırılmış yolu ölç")
    parser.add_argument("--compare", action="store_true", help="hər ikisini ölç və müqayisə et")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=config.CANDIDATES_PER_CATEGORY,
                        help="slot başına namizəd sayı (K)")
    args = parser.parse_args()

    if args.clear:
        print(f"{clear_synthetic()} sintetik sətir silindi")
        return 0

    reference = args.reference or str(sorted(config.IMAGE_DIR.glob("*.jpg"))[0])

    if args.scale:
        added = seed_synthetic(args.scale)
        print(f"{added} sintetik sətir əlavə edildi (source={BENCH_SOURCE!r})")

    try:
        print(f"qarderob   : {wardrobe_size()} sətir")
        print(f"istifadəçi : {args.user}")
        print(f"referens   : {reference}")
        print(f"hədəf      : top-{args.top} < {config.LATENCY_TARGET_SEC} san\n")

        naive_profile: Profile | None = None
        naive_outfits: list = []

        if args.baseline or args.compare:
            naive_profile = Profile("baseline")
            start = time.perf_counter()
            naive_outfits = recommend_naive(args.user, reference, naive_profile, top_k=args.top)
            wall = time.perf_counter() - start
            request = _request_seconds(naive_profile)

            print("=== SADƏLÖVH BASELINE ===")
            print(naive_profile.table())
            # `setup_*` mərhələləri proses-başına xərcdir (uzun ömürlü servisdə
            # bir dəfə). Hədəf `< 2 san` SORĞU-başına vaxta aiddir.
            print(f"\nhazırlıq (proses-başına, bir dəfəlik): "
                  f"{naive_profile.total() - request:.3f} san")
            print(f"SORĞU (hər dəfə)                     : {request:.3f} san "
                  f"({'HƏDƏFDƏN KƏNAR' if request > config.LATENCY_TARGET_SEC else 'hədəf daxilində'})")
            print(f"divar saatı (hər ikisi + ölçülməyən) : {wall:.3f} san")
            print(f"\n=== top-{args.top} outfit ===")
            for rank, (score, combo) in enumerate(naive_outfits, start=1):
                print(f"  {rank}. {score:.4f}  {' + '.join(combo)}")
            if not args.compare:
                return 0
            print()

        if args.fast or args.compare:
            fast_profile = Profile("fast")
            start = time.perf_counter()
            fast_outfits = recommend_fast(
                args.user, reference, fast_profile, top_k=args.top,
                candidates=args.candidates,
            )
            fast_wall = time.perf_counter() - start
            fast_request = _request_seconds(fast_profile)

            print("=== OPTİMALLAŞDIRILMIŞ ===")
            print(fast_profile.table())
            print(f"\nhazırlıq (bir dəfəlik): {fast_profile.total() - fast_request:.3f} san")
            print(f"SORĞU (hər dəfə)      : {fast_request:.3f} san "
                  f"({'HƏDƏFDƏN KƏNAR' if fast_request > config.LATENCY_TARGET_SEC else 'HƏDƏF DAXİLİNDƏ ✓'})")
            print(f"divar saatı           : {fast_wall:.3f} san")
            print(f"\n=== top-{args.top} outfit ===")
            for rank, (score, combo) in enumerate(fast_outfits, start=1):
                print(f"  {rank}. {score:.4f}  {' + '.join(combo)}")

            if args.compare:
                print(f"\n{'=' * 60}\n=== ÖNCƏ / SONRA ===")
                naive_request = _request_seconds(naive_profile)
                rows = [
                    ("DB", "db_wardrobe_n+1", "db_wardrobe_tək"),
                    ("scoring (stil)", "scoring_style_loop", "scoring_matris"),
                    ("slot təyini", "slot_təyini_loop", "slot_təyini_matris"),
                    ("kombinatorika", "kombinatorika_tam", "kombinatorika_topk"),
                ]
                print(f"{'mərhələ':<18}{'öncə (s)':>12}{'sonra (s)':>12}{'sürətlənmə':>14}")
                print("-" * 56)
                for label, old, new in rows:
                    before, after = naive_profile.total(old), fast_profile.total(new)
                    ratio = f"{before / after:.0f}x" if after > 0 else "—"
                    print(f"{label:<18}{before:>12.3f}{after:>12.3f}{ratio:>14}")
                print("-" * 56)
                print(f"{'SORĞU CƏMİ':<18}{naive_request:>12.3f}{fast_request:>12.3f}"
                      f"{naive_request / fast_request:>13.0f}x")
                print(f"\nkombinasiya: {naive_profile.counters.get('yoxlanan_kombinasiya', 0):,}"
                      f" -> {fast_profile.counters.get('yoxlanan_kombinasiya', 0):,}")
                print(f"SQL sorğu  : {naive_profile.counters.get('sql_sorğu', 0):,}"
                      f" -> {fast_profile.counters.get('sql_sorğu', 0):,}")

                # Balları BƏRABƏRLİYƏ görə müqayisə etmək olmaz: sadəlövh yol
                # `np.dot`-u əşya-əşya, sürətli yol tam matris hasili ilə
                # hesablayır — float32-də yığılma sırası fərqli olduğu üçün
                # son bitlərdə fərq normaldır. Sıralama isə eyni olmalıdır.
                naive_combos = [c for _, c in naive_outfits]
                fast_combos = [c for _, c in fast_outfits]
                same_order = naive_combos == fast_combos
                max_delta = max(
                    (abs(a - b) for (a, _), (b, _) in zip(naive_outfits, fast_outfits)),
                    default=0.0,
                )
                print(f"\nNƏTİCƏ EYNİDİRMİ: {'BƏLİ ✓' if same_order else 'XEYR ✗'} "
                      f"(sıralama), maks. bal fərqi {max_delta:.2e}")
                if not same_order:
                    print("  sadəlövh:", naive_combos)
                    print("  sürətli :", fast_combos)
            return 0

        parser.error("bir rejim seç: --baseline | --fast | --compare (--clear ilə təmizlə)")
    finally:
        if args.scale and not args.keep_synthetic:
            print(f"\n[təmizləmə] {clear_synthetic()} sintetik sətir silindi")


if __name__ == "__main__":
    raise SystemExit(main())
