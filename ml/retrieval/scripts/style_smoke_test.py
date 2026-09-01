"""Stil ballandırması smoke test — style_prompts / style_scorer / personal_style / perf.

`scripts/smoke_test.py` ilə eyni üslubda: xarici test kitabxanası YOXDUR
(pytest asılılıq əlavə etmək demək olardı), sadə assert-lər və hesabat.

DB tələb edən testlər DB əlçatmazdırsa ATLANIR (xəta yox) — modelin özü
tələb olunur, çünki stil embedding-ləri onsuz doğrulana bilməz.

İstifadə:
    python -m ml.retrieval.scripts.style_smoke_test
    python -m ml.retrieval.scripts.style_smoke_test --no-db     # yalnız saf testlər
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ml.retrieval import config
from ml.retrieval import perf, personal_style, style_prompts, style_scorer

PASS, FAIL, SKIP = "keçdi", "SINDI", "atlandı"


class Runner:
    """Minimal test qeydiyyatı — ad, nəticə, səbəb."""

    def __init__(self) -> None:
        self.results: list[tuple[str, str, str]] = []

    def run(self, name: str, fn) -> None:
        try:
            outcome = fn()
        except SkipTest as skip:
            self.results.append((name, SKIP, str(skip)))
            print(f"  [{SKIP}] {name} — {skip}")
        except AssertionError as error:
            self.results.append((name, FAIL, str(error)))
            print(f"  [{FAIL}] {name}\n        {error}")
        except Exception as error:  # gözlənilməz xəta da uğursuzluqdur
            detail = f"{type(error).__name__}: {error}"
            self.results.append((name, FAIL, detail))
            print(f"  [{FAIL}] {name}\n        {detail}")
            traceback.print_exc()
        else:
            note = f" — {outcome}" if outcome else ""
            self.results.append((name, PASS, str(outcome or "")))
            print(f"  [{PASS}] {name}{note}")

    def summary(self) -> int:
        passed = sum(1 for _, s, _ in self.results if s == PASS)
        failed = sum(1 for _, s, _ in self.results if s == FAIL)
        skipped = sum(1 for _, s, _ in self.results if s == SKIP)
        print(f"\n{'=' * 62}")
        print(f"{passed} keçdi, {failed} sındı, {skipped} atlandı "
              f"(cəmi {len(self.results)})")
        if failed:
            print("\nSINAN TESTLƏR:")
            for name, status, detail in self.results:
                if status == FAIL:
                    print(f"  - {name}: {detail}")
        return 1 if failed else 0


class SkipTest(Exception):
    """Test şərtləri yoxdur (məs. DB qapalıdır) — uğursuzluq deyil."""


# =========================================================================
# style_prompts
# =========================================================================
def test_prompt_count_and_order():
    prompts = style_prompts.build_prompts()
    expected = len(config.STYLES) * len(config.TEMPLATES)
    assert len(prompts) == expected, f"{len(prompts)} prompt, gözlənilən {expected}"
    # Stil-əsas sıra: ilk T prompt birinci stilə aiddir (reshape buna güvənir)
    first_style = config.STYLES[0]
    for i in range(len(config.TEMPLATES)):
        assert first_style in prompts[i], f"prompt[{i}] {first_style!r} ehtiva etmir"
    assert config.STYLES[1] in prompts[len(config.TEMPLATES)], "stil-əsas sıra pozulub"
    return f"{len(prompts)} prompt, sıra düzgün"


def test_hash_sensitivity():
    base = style_prompts.prompt_hash()
    assert style_prompts.prompt_hash(model_id="başqa/model") != base, "MODEL_ID hash-i dəyişmir"
    assert style_prompts.prompt_hash(styles=[*config.STYLES, "retro"]) != base, \
        "STYLES hash-i dəyişmir"
    assert style_prompts.prompt_hash(templates=config.TEMPLATES[:4]) != base, \
        "TEMPLATES hash-i dəyişmir"
    # Ayırıcı olmasa ["ab","c"] və ["a","bc"] eyni hash verərdi
    assert style_prompts.prompt_hash(styles=["ab", "c"]) != \
        style_prompts.prompt_hash(styles=["a", "bc"]), "ayırıcı işləmir"
    assert style_prompts.prompt_hash() == base, "hash deterministik deyil"
    return f"hash={base}"


def _assert_contract(vectors: np.ndarray, source: str) -> np.ndarray:
    assert vectors.shape == (len(config.STYLES), config.EMB_DIM), \
        f"{source}: shape {vectors.shape}"
    assert vectors.dtype == np.float32, f"{source}: dtype {vectors.dtype}"
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), \
        f"{source}: L2 norm [{norms.min():.4f}, {norms.max():.4f}] — l2_normalize çağırılmayıb?"
    assert np.isfinite(vectors).all(), f"{source}: NaN/Inf var"
    return norms


def test_style_embeddings_contract(embedder):
    """Müqavilə HƏM təzə hesablamada, HƏM keşdə yoxlanılır.

    Yalnız keşi yoxlamaq kifayət deyil: keş artıq mövcud olduğu üçün
    hesablama yolu (`compute_style_embeddings`) heç vaxt işə düşməzdi və
    ondakı normalizasiya səhvi testdən keçərdi. Mutasiya testi bu boşluğu
    məhz belə üzə çıxardı — ona görə burada refresh=True məcburidir.
    """
    tmp = Path(tempfile.mkdtemp(prefix="styla_contract_"))
    try:
        fresh = style_prompts.load_style_embeddings(embedder, refresh=True, cache_dir=tmp)
        assert not fresh.from_cache, "refresh=True keşdən oxudu"
        _assert_contract(fresh.vectors, "təzə hesablama")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    cached = style_prompts.load_style_embeddings(embedder)
    _assert_contract(cached.vectors, "keş")
    assert np.allclose(fresh.vectors, cached.vectors, atol=1e-5), \
        "keşlənmiş vektorlar təzə hesablamadan fərqlidir"
    return f"{cached.vectors.shape} float32, norm≈1 (təzə + keş)"


def test_cache_roundtrip(embedder):
    """Keş yazılır və ikinci çağırışda oxunur (model yenidən işlədilmir)."""
    tmp = Path(tempfile.mkdtemp(prefix="styla_cache_"))
    try:
        first = style_prompts.load_style_embeddings(embedder, cache_dir=tmp)
        assert not first.from_cache, "ilk çağırış keşdən gəlməməli idi"
        assert first.path.exists(), "keş faylı yaranmadı"
        second = style_prompts.load_style_embeddings(embedder, cache_dir=tmp)
        assert second.from_cache, "ikinci çağırış keşdən gəlmədi"
        assert np.array_equal(first.vectors, second.vectors), "keşlənmiş vektorlar fərqlidir"
        return f"{first.path.name}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cache_invalidation_on_templates(embedder):
    """TEMPLATES dəyişəndə fayl adı dəyişir -> köhnə keş istifadə olunmur."""
    tmp = Path(tempfile.mkdtemp(prefix="styla_cache_"))
    try:
        full = style_prompts.load_style_embeddings(embedder, cache_dir=tmp)
        fewer = style_prompts.load_style_embeddings(
            embedder, templates=config.TEMPLATES[:4], cache_dir=tmp
        )
        assert full.path != fewer.path, "fərqli TEMPLATES eyni keş faylına yazır"
        assert not fewer.from_cache, "yeni TEMPLATES köhnə keşdən oxundu"
        assert not np.array_equal(full.vectors, fewer.vectors), \
            "fərqli şablon dəsti eyni vektor verdi"
        return f"{full.path.name} != {fewer.path.name}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cache_corrupted_file(embedder):
    """Korlanmış keş faylı xəta atmır — yenidən qurulur."""
    tmp = Path(tempfile.mkdtemp(prefix="styla_cache_"))
    try:
        first = style_prompts.load_style_embeddings(embedder, cache_dir=tmp)
        first.path.write_bytes(b"bu npz deyil")
        recovered = style_prompts.load_style_embeddings(embedder, cache_dir=tmp)
        assert not recovered.from_cache, "korlanmış fayl keş kimi qəbul edildi"
        assert np.allclose(recovered.vectors, first.vectors, atol=1e-5), \
            "bərpa olunmuş vektorlar fərqlidir"
        return "bərpa olundu"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cache_content_mismatch(embedder):
    """Fayl adı düz, məzmun uyğun deyil -> ikinci qapı tutur."""
    tmp = Path(tempfile.mkdtemp(prefix="styla_cache_"))
    try:
        first = style_prompts.load_style_embeddings(embedder, cache_dir=tmp)
        payload = dict(np.load(first.path, allow_pickle=False))
        payload["styles"] = np.array([*config.STYLES[:-1], "başqa"])
        np.savez(first.path, **payload)
        recovered = style_prompts.load_style_embeddings(embedder, cache_dir=tmp)
        assert not recovered.from_cache, "uyğunsuz məzmun keş kimi qəbul edildi"
        return "metadata yoxlaması işlədi"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_similarity_matrix(embedder):
    style_embs = style_prompts.load_style_embeddings(embedder)
    matrix = style_embs.similarity_matrix()
    size = len(config.STYLES)
    assert matrix.shape == (size, size), f"shape {matrix.shape}"
    assert np.allclose(np.diag(matrix), 1.0, atol=1e-4), "diaqonal 1.0 deyil"
    assert np.allclose(matrix, matrix.T, atol=1e-5), "matris simmetrik deyil"
    clashing = style_prompts.collisions(style_embs.styles, matrix)
    assert not clashing, f"{config.STYLE_COLLISION_MAX}-dan yuxarı cüt var: {clashing}"
    off = matrix[~np.eye(size, dtype=bool)]
    return f"max diaqonaldan kənar {off.max():.3f} < {config.STYLE_COLLISION_MAX}"


def test_collisions_detects():
    """collisions() süni toqquşmanı tutmalıdır (əks halda yalançı 'OK' verərdi)."""
    styles = ["a", "b", "c"]
    matrix = np.array([[1.0, 0.95, 0.2], [0.95, 1.0, 0.3], [0.2, 0.3, 1.0]])
    found = style_prompts.collisions(styles, matrix, max_sim=0.9)
    assert len(found) == 1, f"{len(found)} toqquşma tapıldı, 1 gözlənilirdi"
    assert found[0][:2] == ("a", "b"), f"səhv cüt: {found[0]}"
    assert not style_prompts.collisions(styles, matrix, max_sim=0.99), \
        "hədd yuxarı olanda toqquşma qalmamalıdır"
    return "süni toqquşma tutuldu"


# =========================================================================
# style_scorer
# =========================================================================
def test_score_styles_shapes(scorer, wardrobe):
    result = scorer.score_styles(wardrobe)
    for key in ("styles", "cosine", "centered", "probs", "logit_scale", "centering"):
        assert key in result, f"{key!r} açarı yoxdur"
    n, s = len(wardrobe), len(config.STYLES)
    assert result["cosine"].shape == (n, s), f"cosine {result['cosine'].shape}"
    assert result["probs"].shape == (n, s), f"probs {result['probs'].shape}"
    assert result["styles"] == config.STYLES, "stil siyahısı uyğun deyil"
    return f"cosine/probs {result['cosine'].shape}"


def test_cosine_matches_manual(scorer, wardrobe):
    """Matris hasili əl ilə hesablanmış skalar hasillə üst-üstə düşməlidir."""
    result = scorer.score_styles(wardrobe[:5])
    style_vectors = scorer.style_embs.vectors
    for i in range(5):
        for j in range(len(config.STYLES)):
            manual = float(np.dot(wardrobe[i], style_vectors[j]))
            assert abs(manual - result["cosine"][i, j]) < 1e-5, \
                f"[{i},{j}] {manual} != {result['cosine'][i, j]}"
    return "5x8 dəyər əl hesabı ilə uyğundur"


def test_probs_sum_to_one(scorer, wardrobe):
    result = scorer.score_styles(wardrobe)
    sums = result["probs"].sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-5), f"sətir cəmləri [{sums.min()}, {sums.max()}]"
    assert (result["probs"] >= 0).all(), "mənfi ehtimal var"
    return f"bütün sətirlər 1.0 (±{abs(sums - 1.0).max():.1e})"


def test_centering_zeroes_column_means(scorer, wardrobe):
    result = scorer.score_styles(wardrobe, centering=True)
    assert result["centering"], "centering tətbiq olunmadı"
    means = result["centered"].mean(axis=0)
    assert np.abs(means).max() < 1e-5, f"sütun ortası sıfır deyil: {np.abs(means).max()}"
    # Xam cosine TOXUNULMAZ qalmalıdır
    manual = wardrobe @ scorer.style_embs.vectors.T
    assert np.allclose(result["cosine"], manual, atol=1e-6), "xam cosine dəyişdirilib"
    return f"maks. sütun ortası {np.abs(means).max():.1e}"


def test_centering_off_is_identity(scorer, wardrobe):
    result = scorer.score_styles(wardrobe, centering=False)
    assert not result["centering"], "centering sönülü olmalı idi"
    assert np.array_equal(result["centered"], result["cosine"]), \
        "centering sönülüdürsə centered == cosine olmalıdır"
    return "centered == cosine"


def test_centering_skipped_for_single_item(scorer, wardrobe):
    """N=1 üçün centering sətri sıfırlayardı -> avtomatik atlanmalıdır."""
    result = scorer.score_styles(wardrobe[:1], centering=True)
    assert not result["centering"], "N=1 üçün centering atlanmadı"
    assert not np.allclose(result["probs"], 1.0 / len(config.STYLES)), \
        "probs bərabər paylanmaya çevrilib — centering sıfırlama baş verib"
    return "atlandı, probs mənalı qaldı"


def test_softmax_stability():
    """Böyük logit-lərdə overflow olmamalıdır (logit_scale ~100 ilə real risk)."""
    logits = np.array([[1000.0, 999.0, -1000.0]])
    probs = style_scorer.softmax(logits)
    assert np.isfinite(probs).all(), "softmax NaN/Inf verdi"
    assert abs(probs.sum() - 1.0) < 1e-9, f"cəm {probs.sum()}"
    assert probs[0, 0] > probs[0, 1] > probs[0, 2], "sıralama pozulub"
    return "1e3 logit-də stabil"


def test_rank_by_style_uses_raw_cosine(scorer, wardrobe):
    """Item-lərarası sıralama XAM cosine ilə olmalıdır, probs ilə yox."""
    result = scorer.score_styles(wardrobe)
    style = config.STYLES[1]
    ranking = scorer.rank_by_style(result, style)
    column = result["cosine"][:, config.STYLES.index(style)]
    expected = list(np.argsort(-column))
    assert [i for i, _ in ranking] == expected, "sıralama xam cosine ilə uyğun deyil"
    scores = [s for _, s in ranking]
    assert scores == sorted(scores, reverse=True), "azalan sıra pozulub"
    return f"'{style}' üzrə {len(ranking)} item"


def test_score_styles_rejects_bad_dim(scorer):
    try:
        scorer.score_styles(np.zeros((3, 128), dtype=np.float32))
    except ValueError:
        return "səhv ölçü rədd edildi"
    raise AssertionError("səhv ölçülü embedding qəbul edildi")


# =========================================================================
# personal_style
# =========================================================================
def test_make_ref_id():
    first = personal_style.make_ref_id("u1", "a/b.jpg")
    assert first == personal_style.make_ref_id("u1", "a/b.jpg"), "deterministik deyil"
    assert first != personal_style.make_ref_id("u2", "a/b.jpg"), "user_id-dən asılı deyil"
    assert first != personal_style.make_ref_id("u1", "a/c.jpg"), "yoldan asılı deyil"
    return first


def test_aggregate_max_and_mean_top2():
    items = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    refs = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]], dtype=np.float32)
    sims = items @ refs.T  # [[1.0, 0.6, 0.0], [0.0, 0.8, 1.0]]

    got_max = personal_style.aggregate_similarity(items, refs, "max")
    assert np.allclose(got_max, sims.max(axis=1)), f"max səhv: {got_max}"

    got_top2 = personal_style.aggregate_similarity(items, refs, "mean_top2")
    expected = np.sort(sims, axis=1)[:, -2:].mean(axis=1)
    assert np.allclose(got_top2, expected), f"mean_top2 səhv: {got_top2} != {expected}"
    # mean_top2 həmişə max-dan kiçik və ya bərabərdir
    assert (got_top2 <= got_max + 1e-6).all(), "mean_top2 max-ı keçdi"
    return f"max={got_max.round(2).tolist()} top2={got_top2.round(2).tolist()}"


def test_aggregate_single_ref_equivalence():
    """Tək referansda mean_top2 max-a bərabər olmalıdır (top-2 yoxdur)."""
    items = np.random.default_rng(0).standard_normal((4, 8)).astype(np.float32)
    refs = np.random.default_rng(1).standard_normal((1, 8)).astype(np.float32)
    got_max = personal_style.aggregate_similarity(items, refs, "max")
    got_top2 = personal_style.aggregate_similarity(items, refs, "mean_top2")
    assert np.allclose(got_max, got_top2), f"{got_max} != {got_top2}"
    return "tək referansda eynidir"


def test_aggregate_empty_refs():
    items = np.ones((3, config.EMB_DIM), dtype=np.float32)
    empty = np.zeros((0, config.EMB_DIM), dtype=np.float32)
    scores = personal_style.aggregate_similarity(items, empty, "max")
    assert scores.shape == (3,), f"shape {scores.shape}"
    assert np.array_equal(scores, np.zeros(3)), "referansı olmayan istifadəçi sıfır almalıdır"
    return "sıfır massiv, istisna yox"


def test_aggregate_rejects_unknown_agg():
    items = np.ones((2, 4), dtype=np.float32)
    refs = np.ones((2, 4), dtype=np.float32)
    try:
        personal_style.aggregate_similarity(items, refs, "median")
    except ValueError:
        return "naməlum agg rədd edildi"
    raise AssertionError("naməlum aqreqasiya qəbul edildi")


def test_normalize_for_fusion():
    scores = np.array([0.4, 0.5, 0.7, 0.9, 1.2], dtype=np.float32)
    out = personal_style.normalize_for_fusion(scores, 0.5, 0.9)
    assert out.min() >= 0.0 and out.max() <= 1.0, f"[0,1] xaricində: {out}"
    assert out[0] == 0.0, "aşağı hədd altındakı 0-a kəsilməlidir"
    assert out[-1] == 1.0, "yuxarı hədd üstündəki 1-ə kəsilməlidir"
    assert abs(out[2] - 0.5) < 1e-6, f"orta nöqtə 0.5 olmalıdır, alındı {out[2]}"
    try:
        personal_style.normalize_for_fusion(scores, 0.9, 0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("high <= low qəbul edildi")
    return "kəsim və miqyas düzgün"


def test_as_array_accepts_both():
    """pgvector versiyasından asılı olaraq Vector və ya ndarray gəlir."""
    from pgvector import Vector

    plain = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert np.array_equal(personal_style._as_array(plain), plain), "ndarray keçmədi"
    converted = personal_style._as_array(Vector([1.0, 2.0, 3.0]))
    assert np.array_equal(converted, plain), f"Vector çevrilmədi: {converted}"
    assert converted.dtype == np.float32, f"dtype {converted.dtype}"
    return "Vector və ndarray dəstəklənir"


# --- DB tələb edən -------------------------------------------------------
def _open_personal(embedder):
    try:
        return personal_style.PersonalStyle(embedder=embedder)
    except Exception as error:
        raise SkipTest(f"DB əlçatmazdır ({type(error).__name__})")


def test_db_refs_roundtrip(embedder, images):
    user = "__smoke_test_user__"
    with _open_personal(embedder) as store:
        store.delete_refs(user)
        try:
            written = store.add_style_refs(user, images[:3])
            assert written == 3, f"{written} sətir yazıldı, 3 gözlənilirdi"
            assert store.count(user) == 3, f"count {store.count(user)}"

            ref_ids, refs = store.get_refs(user)
            assert refs.shape == (3, config.EMB_DIM), f"shape {refs.shape}"
            assert refs.dtype == np.float32, f"dtype {refs.dtype}"
            norms = np.linalg.norm(refs, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-3), f"norm [{norms.min()}, {norms.max()}]"

            # Upsert: eyni şəkillər yenidən -> sətir sayı artmamalıdır
            store.add_style_refs(user, images[:3])
            assert store.count(user) == 3, f"upsert təkrar sətir yaratdı: {store.count(user)}"

            # model_ver filtri: uyğunsuz versiya heç nə qaytarmamalıdır
            _, mismatched = store.get_refs(user, model_ver="yoxdur-v99")
            assert len(mismatched) == 0, f"uyğunsuz model_ver {len(mismatched)} sətir verdi"
            return f"{len(ref_ids)} referans, upsert və model_ver filtri işləyir"
        finally:
            store.delete_refs(user)


def test_db_personal_score_generalizes(embedder, images):
    """Referans verilməyən, amma eyni tipli şəkil yüksək bal almalıdır."""
    user = "__smoke_test_user__"
    with _open_personal(embedder) as store:
        store.delete_refs(user)
        try:
            store.add_style_refs(user, images[:2])
            probe = embedder.embed_images(images[:3])
            scores = store.personal_score(probe, user)
            assert scores.shape == (3,), f"shape {scores.shape}"
            # İlk ikisi referansların ÖZÜdür -> ~1.0
            assert scores[0] > 0.99 and scores[1] > 0.99, \
                f"referansın özü 1.0 vermədi: {scores[:2]}"
            assert 0.0 <= scores[2] <= 1.0, f"bal diapazondan kənar: {scores[2]}"
            # Referansı olmayan istifadəçi -> sıfır
            zeros = store.personal_score(probe, "__yoxdur__")
            assert np.array_equal(zeros, np.zeros(3)), f"boş istifadəçi {zeros} verdi"
            return f"öz={scores[0]:.3f} yad={scores[2]:.3f}"
        finally:
            store.delete_refs(user)


def test_db_no_ann_index(embedder):
    """user_style_refs-də vektor indeksi OLMAMALIDIR (qəsdən qərar)."""
    with _open_personal(embedder) as store:
        with store.conn.cursor() as cur:
            cur.execute(
                """SELECT indexname FROM pg_indexes
                   WHERE tablename = %s AND (indexdef ILIKE %s OR indexdef ILIKE %s)""",
                (store.table, "%USING ivfflat%", "%USING hnsw%"),
            )
            ann = [r[0] for r in cur.fetchall()]
        assert not ann, f"ANN indeks tapıldı: {ann}"

        with store.conn.cursor() as cur:
            cur.execute(
                "SELECT indexname FROM pg_indexes WHERE tablename = %s", (store.table,)
            )
            names = [r[0] for r in cur.fetchall()]
        assert f"{store.table}_user_id_idx" in names, f"user_id indeksi yoxdur: {names}"
        return f"{len(names)} B-tree indeks, ANN yoxdur"


# =========================================================================
# perf
# =========================================================================
def test_profile_records():
    profile = perf.Profile("test")
    with profile.stage("a"):
        pass
    with profile.stage("a"):
        pass
    with profile.stage("b"):
        pass
    assert profile.calls("a") == 2, f"a: {profile.calls('a')} çağırış"
    assert profile.calls("b") == 1, f"b: {profile.calls('b')} çağırış"
    assert profile.total() >= profile.total("a"), "cəm mərhələ cəmindən kiçikdir"
    profile.count("sorğu", 5)
    profile.count("sorğu")
    assert profile.counters["sorğu"] == 6, f"sayğac {profile.counters['sorğu']}"
    assert "CƏMİ" in profile.table(), "cədvəldə CƏMİ yoxdur"
    return "mərhələ, çağırış sayı və sayğac"


def test_profile_records_on_exception():
    """İstisna baş versə də müddət yazılmalıdır (finally)."""
    profile = perf.Profile()
    try:
        with profile.stage("partlayan"):
            raise RuntimeError("qəsdən")
    except RuntimeError:
        pass
    assert profile.calls("partlayan") == 1, "istisna zamanı müddət yazılmadı"
    return "finally işləyir"


def test_timed_helper():
    with perf.timed() as holder:
        sum(range(1000))
    assert holder[0] > 0, f"müddət {holder[0]}"
    return f"{holder[0] * 1e6:.0f} µs"


def test_coarse_to_slot():
    mapping = perf.coarse_to_slot()
    total = sum(len(v) for v in config.OUTFIT_SLOTS.values())
    assert len(mapping) == total, f"{len(mapping)} xəritələmə, {total} gözlənilirdi"
    for slot, labels in config.OUTFIT_SLOTS.items():
        for label in labels:
            assert mapping[label] == slot, f"{label} -> {mapping[label]}, {slot} gözlənilirdi"
    # Slotlar kəsişməməlidir — bir kateqoriya iki slota düşsə nəticə qeyri-müəyyən olar
    assert len(mapping) == len({l for ls in config.OUTFIT_SLOTS.values() for l in ls}), \
        "slotlar arasında təkrarlanan kateqoriya var"
    return f"{len(mapping)} kateqoriya -> {len(config.OUTFIT_SLOTS)} slot"


def test_topk_cut_is_lossless():
    """Additiv balda top-K kəsimi düzgün cavabı ata bilməz.

    Kobud qüvvə ilə tam Dekart hasilini hesablayıb top-K kəsimi ilə
    müqayisə edirik — perf.recommend_fast-ın əsaslandığı fərziyyə budur.
    """
    rng = np.random.default_rng(7)
    top_scores = rng.standard_normal(40)
    bottom_scores = rng.standard_normal(35)
    shoe_scores = rng.standard_normal(30)
    want = 5
    k = config.CANDIDATES_PER_CATEGORY

    brute = sorted(
        (t + b + s for t in top_scores for b in bottom_scores for s in shoe_scores),
        reverse=True,
    )[:want]

    def top_k(values):
        return np.sort(values)[::-1][:k]

    cut = sorted(
        (t + b + s
         for t in top_k(top_scores)
         for b in top_k(bottom_scores)
         for s in top_k(shoe_scores)),
        reverse=True,
    )[:want]

    assert np.allclose(brute, cut), f"kəsim düzgün cavabı itirdi:\n  {brute}\n  {cut}"
    return f"{len(top_scores) * len(bottom_scores) * len(shoe_scores):,} -> {k ** 3:,} kombinasiya, eyni top-{want}"


def test_score_outfit_additive():
    style_row = np.array([0.1, 0.3, 0.2], dtype=np.float32)
    assert abs(perf.score_outfit(style_row, 0.5) - 0.8) < 1e-6, "ən güclü stil + şəxsi bal"
    return "0.3 + 0.5 = 0.8"


def test_naive_and_fast_agree(images):
    """Ən vacib test: iki implementasiya EYNİ top-5 verməlidir."""
    try:
        size = perf.wardrobe_size()
    except Exception as error:
        raise SkipTest(f"DB əlçatmazdır ({type(error).__name__})")
    if size == 0:
        raise SkipTest("qarderob boşdur -> python -m ml.retrieval.ingest")

    user = "__smoke_test_user__"
    reference = str(images[0])
    with personal_style.PersonalStyle() as store:
        store.delete_refs(user)
        store.add_style_refs(user, images[:2])
    try:
        naive = perf.recommend_naive(user, reference, perf.Profile(), top_k=5)
        fast = perf.recommend_fast(user, reference, perf.Profile(), top_k=5)
        assert [c for _, c in naive] == [c for _, c in fast], (
            f"top-5 fərqlidir:\n  sadəlövh: {[c for _, c in naive]}\n"
            f"  sürətli : {[c for _, c in fast]}"
        )
        deltas = [abs(a - b) for (a, _), (b, _) in zip(naive, fast)]
        assert max(deltas) < 1e-4, f"bal fərqi çox böyükdür: {max(deltas)}"
        return f"top-5 eyni, maks. bal fərqi {max(deltas):.2e}"
    finally:
        with personal_style.PersonalStyle() as store:
            store.delete_refs(user)


def test_synthetic_cleanup():
    """--scale sətirləri tam təmizlənməlidir (real qarderob toxunulmamalıdır)."""
    try:
        before = perf.wardrobe_size()
    except Exception as error:
        raise SkipTest(f"DB əlçatmazdır ({type(error).__name__})")
    perf.seed_synthetic(25)
    assert perf.wardrobe_size() == before + 25, "sintetik sətirlər əlavə olunmadı"
    removed = perf.clear_synthetic()
    assert removed == 25, f"{removed} silindi, 25 gözlənilirdi"
    assert perf.wardrobe_size() == before, "təmizləmədən sonra sətir sayı bərpa olunmadı"
    return f"{before} -> {before + 25} -> {before}"


# =========================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="Stil ballandırması smoke test")
    parser.add_argument("--no-db", action="store_true", help="DB testlərini atla")
    args = parser.parse_args()

    from ml.retrieval.embedder import FashionCLIPEmbedder

    print("Model yüklənir (bir dəfə, bütün testlər üçün)...")
    embedder = FashionCLIPEmbedder()
    embedder._ensure_loaded()

    images = sorted(config.IMAGE_DIR.glob("*.jpg"))[:5]
    have_images = len(images) >= 3
    scorer = style_scorer.StyleScorer(embedder=embedder)
    try:
        _, wardrobe = style_scorer.load_wardrobe()
    except FileNotFoundError:
        wardrobe = None

    runner = Runner()

    print("\n--- style_prompts ---")
    runner.run("prompt sayı və sırası", test_prompt_count_and_order)
    runner.run("hash həssaslığı", test_hash_sensitivity)
    runner.run("embedding müqaviləsi", lambda: test_style_embeddings_contract(embedder))
    runner.run("keş roundtrip", lambda: test_cache_roundtrip(embedder))
    runner.run("keş: TEMPLATES dəyişdi", lambda: test_cache_invalidation_on_templates(embedder))
    runner.run("keş: korlanmış fayl", lambda: test_cache_corrupted_file(embedder))
    runner.run("keş: məzmun uyğunsuzluğu", lambda: test_cache_content_mismatch(embedder))
    runner.run("8x8 oxşarlıq matrisi", lambda: test_similarity_matrix(embedder))
    runner.run("collisions() toqquşmanı tutur", test_collisions_detects)

    print("\n--- style_scorer ---")
    if wardrobe is None:
        for name in ("shape-lər", "əl hesabı", "probs cəmi", "centering", "centering sönülü",
                     "N=1 centering", "rank_by_style"):
            runner.run(name, lambda: (_ for _ in ()).throw(
                SkipTest("qarderob yoxdur -> python -m ml.retrieval.ingest")))
    else:
        runner.run("shape-lər və açarlar", lambda: test_score_styles_shapes(scorer, wardrobe))
        runner.run("cosine == əl hesabı", lambda: test_cosine_matches_manual(scorer, wardrobe))
        runner.run("probs sətir cəmi = 1", lambda: test_probs_sum_to_one(scorer, wardrobe))
        runner.run("centering sütun ortasını sıfırlayır",
                   lambda: test_centering_zeroes_column_means(scorer, wardrobe))
        runner.run("centering sönülü = eynilik",
                   lambda: test_centering_off_is_identity(scorer, wardrobe))
        runner.run("N=1 üçün centering atlanır",
                   lambda: test_centering_skipped_for_single_item(scorer, wardrobe))
        runner.run("rank_by_style xam cosine işlədir",
                   lambda: test_rank_by_style_uses_raw_cosine(scorer, wardrobe))
    runner.run("softmax stabilliyi", test_softmax_stability)
    runner.run("səhv ölçü rədd edilir", lambda: test_score_styles_rejects_bad_dim(scorer))

    print("\n--- personal_style ---")
    runner.run("make_ref_id determinizmi", test_make_ref_id)
    runner.run("max və mean_top2 düzgünlüyü", test_aggregate_max_and_mean_top2)
    runner.run("tək referansda ekvivalentlik", test_aggregate_single_ref_equivalence)
    runner.run("boş referans -> sıfır", test_aggregate_empty_refs)
    runner.run("naməlum agg rədd edilir", test_aggregate_rejects_unknown_agg)
    runner.run("normalize_for_fusion", test_normalize_for_fusion)
    runner.run("_as_array Vector/ndarray", test_as_array_accepts_both)

    print("\n--- perf ---")
    runner.run("Profile qeydiyyatı", test_profile_records)
    runner.run("Profile istisna zamanı", test_profile_records_on_exception)
    runner.run("timed() köməkçisi", test_timed_helper)
    runner.run("slot xəritələməsi", test_coarse_to_slot)
    runner.run("top-K kəsimi itkisizdir", test_topk_cut_is_lossless)
    runner.run("score_outfit additivliyi", test_score_outfit_additive)

    print("\n--- DB (pgvector) ---")
    if args.no_db:
        for name in ("referans roundtrip", "şəxsi bal ümumiləşməsi", "ANN indeks yoxdur",
                     "sadəlövh == sürətli", "sintetik təmizləmə"):
            runner.run(name, lambda: (_ for _ in ()).throw(SkipTest("--no-db")))
    elif not have_images:
        for name in ("referans roundtrip", "şəxsi bal ümumiləşməsi", "sadəlövh == sürətli"):
            runner.run(name, lambda: (_ for _ in ()).throw(SkipTest("şəkil yoxdur")))
        runner.run("ANN indeks yoxdur", lambda: test_db_no_ann_index(embedder))
        runner.run("sintetik təmizləmə", test_synthetic_cleanup)
    else:
        runner.run("referans roundtrip", lambda: test_db_refs_roundtrip(embedder, images))
        runner.run("şəxsi bal ümumiləşməsi",
                   lambda: test_db_personal_score_generalizes(embedder, images))
        runner.run("ANN indeks yoxdur", lambda: test_db_no_ann_index(embedder))
        runner.run("sadəlövh == sürətli", lambda: test_naive_and_fast_agree(images))
        runner.run("sintetik təmizləmə", test_synthetic_cleanup)

    return runner.summary()


if __name__ == "__main__":
    raise SystemExit(main())
