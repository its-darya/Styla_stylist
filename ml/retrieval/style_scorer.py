"""Zero-shot stil ballandırması.

`style_prompts.py`-ın [8, 512] stil matrisi ilə şəkil embedding-lərini
tutuşdurur. Hər ikisi L2-normalized olduğu üçün skalar hasil birbaşa
cosine similarity-dir:

    cosine = img_embs @ style_embs.T        # [N, 8]

İKİ BAL, İKİ FƏRQLİ İSTİFADƏ (qarışdırma!):
    probs  — BİR item daxilində stilləri sıralamaq üçün.
             Softmax sətir üzrə normalize edir, yəni "bu köynək casual-dan
             çox streetwear-dir" sualına cavab verir. İki item-in probs-larını
             müqayisə etmək mənasızdır — hər sətir ayrıca 1-ə toplanır.
    cosine — İKİ item-i EYNİ stil üzrə müqayisə etmək üçün.
             "Hansı köynək daha çox formal-dır?" sualına yalnız xam cosine
             cavab verə bilər, çünki o, ümumi (item-lərarası) miqyasdadır.

CENTERING (`config.STYLE_CENTERING`):
    CLIP mətn embedding-ləri dar konusda yerləşir və "casual" maqnit sinifdir —
    demək olar hər şey ona ən yaxın çıxır. Sütun (stil) üzrə orta çıxılanda
    həmin sabit meyl yox olur və item-lər arasındakı NİSBİ fərq görünür:

        scores -= scores.mean(axis=0, keepdims=True)

    DİQQƏT: bu, N item-in HAMISI üzrə hesablanır — yəni bal artıq mütləq deyil,
    verilmiş dəstə (batch) daxilində nisbidir. Tək şəkil üçün mənasızdır və
    avtomatik atlanır. Dəstə dəyişsə ballar da dəyişir.

MODALITY GAP: buradakı ballar şəkil↔MƏTN oxşarlığıdır (tipik 0.15-0.35) və
`personal_style.py`-ın şəkil↔ŞƏKİL balları (tipik 0.5-0.9) ilə BİRBAŞA
müqayisə edilə BİLMƏZ.

İstifadə:
    python -m ml.retrieval.style_scorer --limit 10
    python -m ml.retrieval.style_scorer --compare      # centering öncə/sonra
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):  # birbaşa `python style_scorer.py` işlədiləndə
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.retrieval import config
from ml.retrieval.embedder import FashionCLIPEmbedder
from ml.retrieval.style_prompts import StyleEmbeddings, load_style_embeddings


def softmax(logits: np.ndarray) -> np.ndarray:
    """Sətir üzrə ədədi sabit softmax."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


class StyleScorer:
    """Şəkil embedding-lərini 8 stil üzrə ballandırır.

    Stil matrisi keşdən gəlir — bu sinif model YÜKLƏMİR (embedder yalnız
    `score_images()` çağırılanda və ya keş boş olanda lazımdır).
    """

    def __init__(
        self,
        style_embs: StyleEmbeddings | None = None,
        embedder: FashionCLIPEmbedder | None = None,
        centering: bool | None = None,
    ) -> None:
        self._embedder = embedder
        self.style_embs = style_embs or load_style_embeddings(embedder)
        self.centering = config.STYLE_CENTERING if centering is None else centering
        self._logit_scale: float | None = None

    @property
    def embedder(self) -> FashionCLIPEmbedder:
        if self._embedder is None:
            self._embedder = FashionCLIPEmbedder()
        return self._embedder

    @property
    def styles(self) -> list[str]:
        return self.style_embs.styles

    @property
    def logit_scale(self) -> float:
        """CLIP-in öyrədilmiş `logit_scale.exp()` — softmax temperaturu.

        config-dəki `CATEGORY_LOGIT_SCALE = 100.0` təxmini dəyərdir; burada
        modelin ÖZ öyrədilmiş parametrindən istifadə edirik. Model yüklü
        deyilsə yüklənir (bu, ilk çağırışda ~5-10 s-dir).
        """
        if self._logit_scale is None:
            self._logit_scale = float(self.embedder.model.logit_scale.exp().item())
        return self._logit_scale

    # --- əsas ------------------------------------------------------------
    def score_styles(
        self,
        img_embs: np.ndarray,
        centering: bool | None = None,
        logit_scale: float | None = None,
    ) -> dict[str, Any]:
        """[N, EMB_DIM] -> stil balları.

        Qaytarır:
            styles      : list[str], uzunluq S
            cosine      : [N, S] XAM cosine — item-lərarası müqayisə üçün
            centered    : [N, S] sütun ortası çıxılmış (centering sönülüdürsə
                          `cosine` ilə eynidir)
            probs       : [N, S] softmax — bir item daxilində sıralama üçün
            logit_scale : softmax temperaturu
            centering   : centering həqiqətən tətbiq olundumu
        """
        img_embs = np.asarray(img_embs, dtype=np.float32)
        if img_embs.ndim == 1:
            img_embs = img_embs.reshape(1, -1)
        if img_embs.shape[1] != config.EMB_DIM:
            raise ValueError(
                f"Şəkil embedding-i [N, {config.EMB_DIM}] olmalıdır, alındı {img_embs.shape}"
            )

        cosine = img_embs @ self.style_embs.vectors.T  # [N, S]

        use_centering = self.centering if centering is None else centering
        if use_centering and len(cosine) < 2:
            # Bir sətrin öz ortasını çıxmaq bütün sətri sıfırlayır -> softmax
            # bərabər paylanma verir. Səssiz yanlış nəticə əvəzinə atlayırıq.
            print("[xəbərdarlıq] centering üçün ən azı 2 item lazımdır — atlanır")
            use_centering = False

        centered = cosine - cosine.mean(axis=0, keepdims=True) if use_centering else cosine

        scale = self.logit_scale if logit_scale is None else logit_scale
        probs = softmax(centered.astype(np.float64) * scale).astype(np.float32)

        return {
            "styles": list(self.styles),
            "cosine": cosine,
            "centered": centered,
            "probs": probs,
            "logit_scale": scale,
            "centering": use_centering,
        }

    def score_images(self, paths: Sequence[str | Path], **kwargs) -> dict[str, Any]:
        """Şəkil yollarını embed edib ballandırır."""
        return self.score_styles(self.embedder.embed_images(list(paths)), **kwargs)

    # --- köməkçi ---------------------------------------------------------
    def top_styles(self, result: dict[str, Any], n: int = 1) -> list[list[tuple[str, float]]]:
        """Hər item üçün ən güclü n stil (probs-a görə, azalan)."""
        styles, probs = result["styles"], result["probs"]
        order = np.argsort(-probs, axis=1)[:, :n]
        return [
            [(styles[j], float(probs[i, j])) for j in row]
            for i, row in enumerate(order)
        ]

    def rank_by_style(
        self, result: dict[str, Any], style: str, top_k: int | None = None
    ) -> list[tuple[int, float]]:
        """Item-ləri VERİLMİŞ stil üzrə sıralayır -> [(item indeksi, xam cosine)].

        Qəsdən XAM cosine işlədilir: bu, item-lərarası müqayisədir, probs yox.
        """
        column = result["styles"].index(style)
        scores = result["cosine"][:, column]
        order = np.argsort(-scores)
        if top_k is not None:
            order = order[:top_k]
        return [(int(i), float(scores[i])) for i in order]


# --- formatlaşdırma -------------------------------------------------------
def format_scores(
    result: dict[str, Any],
    ids: Sequence[str],
    matrix_key: str = "probs",
    limit: int | None = None,
    id_width: int = 12,
) -> str:
    """Item × stil cədvəli."""
    styles, matrix = result["styles"], result[matrix_key]
    rows = range(len(matrix) if limit is None else min(limit, len(matrix)))
    header = f"{'item':<{id_width}}" + "".join(f"{s[:7]:>8}" for s in styles) + "   ← ən güclü"
    lines = [header, "-" * len(header)]
    for i in rows:
        best = int(np.argmax(matrix[i]))
        cells = "".join(f"{matrix[i, j]:>8.3f}" for j in range(len(styles)))
        lines.append(f"{str(ids[i])[:id_width - 1]:<{id_width}}{cells}   {styles[best]}")
    return "\n".join(lines)


def style_distribution(result: dict[str, Any]) -> dict[str, int]:
    """Hansı stil neçə item-də qalib gəlib — maqnit sinfi görmək üçün."""
    styles, probs = result["styles"], result["probs"]
    winners = np.argmax(probs, axis=1)
    counts = {s: 0 for s in styles}
    for w in winners:
        counts[styles[int(w)]] += 1
    return counts


def format_distribution(before: dict[str, int], after: dict[str, int], total: int) -> str:
    """Centering öncə/sonra qalib stil paylanması."""
    lines = [
        f"{'stil':<12}{'öncə':>8}{'sonra':>8}   fərq",
        "-" * 44,
    ]
    for style in before:
        b, a = before[style], after[style]
        delta = a - b
        bar = ("+" if delta > 0 else "") + str(delta) if delta else "·"
        lines.append(f"{style:<12}{b:>8}{a:>8}   {bar}")
    lines.append("-" * 44)
    lines.append(f"{'CƏMİ':<12}{total:>8}{total:>8}")
    return "\n".join(lines)


def load_wardrobe(limit: int | None = None) -> tuple[list[str], np.ndarray]:
    """Qarderobu disk artefaktlarından oxuyur — model YÜKLƏNMİR.

    `NumpyStore` onsuz da `.npy` + `ids.json` cütünü oxuyur və müqaviləni
    (`validate_vectors`) yoxlayır — öz oxucumuzu yazmırıq.
    """
    from ml.retrieval.store.numpy_store import NumpyStore

    store = NumpyStore()
    if store.count() == 0:
        raise FileNotFoundError(
            f"{config.EMB_PATH} boşdur/yoxdur -> əvvəlcə `python -m ml.retrieval.ingest`"
        )
    ids, vectors = store.ids, store.vectors
    if limit is not None:
        ids, vectors = ids[:limit], vectors[:limit]
    return list(ids), np.asarray(vectors, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Zero-shot stil ballandırması")
    parser.add_argument("--image", action="append", default=[], help="şəkil yolu (təkrarlana bilər)")
    parser.add_argument("--limit", type=int, default=None, help="qarderobdan neçə item")
    parser.add_argument("--show", type=int, default=12, help="cədvəldə neçə sətir göstərilsin")
    parser.add_argument("--compare", action="store_true",
                        help="centering öncə/sonra müqayisəsi")
    parser.add_argument("--rank-by", default=None,
                        help="item-ləri bu stil üzrə sırala (xam cosine)")
    parser.add_argument("--no-centering", action="store_true")
    args = parser.parse_args()

    scorer = StyleScorer(centering=not args.no_centering)

    if args.image:
        ids = [Path(p).stem for p in args.image]
        embeddings = scorer.embedder.embed_images(args.image)
    else:
        ids, embeddings = load_wardrobe(args.limit)

    print(f"item sayı   : {len(ids)}")
    print(f"stil matrisi: {scorer.style_embs.vectors.shape} "
          f"(prompt_hash={scorer.style_embs.prompt_hash})")
    print(f"logit_scale : {scorer.logit_scale:.4f}  "
          f"(config.CATEGORY_LOGIT_SCALE = {config.CATEGORY_LOGIT_SCALE} — təxmini dəyər)")

    if args.compare:
        without = scorer.score_styles(embeddings, centering=False)
        with_centering = scorer.score_styles(embeddings, centering=True)

        print(f"\n=== XAM cosine diapazonu (stil sütunları üzrə orta) ===")
        column_means = without["cosine"].mean(axis=0)
        for style, mean in zip(without["styles"], column_means):
            print(f"  {style:<12} {mean:.4f}")
        print(f"  cosine min={without['cosine'].min():.4f} "
              f"max={without['cosine'].max():.4f} — şəkil↔MƏTN diapazonu")

        print(f"\n=== CENTERING SÖNÜLÜ — probs (ilk {args.show}) ===")
        print(format_scores(without, ids, "probs", limit=args.show))
        print(f"\n=== CENTERING AÇIQ — probs (ilk {args.show}) ===")
        print(format_scores(with_centering, ids, "probs", limit=args.show))

        before = style_distribution(without)
        after = style_distribution(with_centering)
        print(f"\n=== Qalib stil paylanması ({len(ids)} item) ===")
        print(format_distribution(before, after, len(ids)))

        nonzero_before = sum(1 for v in before.values() if v)
        nonzero_after = sum(1 for v in after.values() if v)
        print(f"\nİstifadə olunan stil sayı: {nonzero_before}/8 -> {nonzero_after}/8")
        changed = int((np.argmax(without["probs"], axis=1)
                       != np.argmax(with_centering["probs"], axis=1)).sum())
        print(f"Qalib stili dəyişən item: {changed}/{len(ids)}")
        return 0

    result = scorer.score_styles(embeddings)
    print(f"centering   : {'AÇIQ' if result['centering'] else 'SÖNÜLÜ'}\n")

    if args.rank_by:
        if args.rank_by not in result["styles"]:
            parser.error(f"naməlum stil: {args.rank_by!r} (mövcud: {result['styles']})")
        print(f"=== '{args.rank_by}' üzrə sıralama (XAM cosine — item-lərarası) ===")
        for rank, (index, score) in enumerate(
            scorer.rank_by_style(result, args.rank_by, args.show), start=1
        ):
            print(f"  {rank:>2}. {ids[index]:<14} {score:.4f}")
        return 0

    print(f"=== probs (bir item daxilində sıralama üçün) ===")
    print(format_scores(result, ids, "probs", limit=args.show))
    print(f"\n=== xam cosine (item-lərarası müqayisə üçün) ===")
    print(format_scores(result, ids, "cosine", limit=args.show))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
