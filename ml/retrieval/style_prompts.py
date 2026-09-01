"""Stil promptları və stil embedding keşi.

Hər stil üçün `config.TEMPLATES`-dəki 5 şablon ayrıca embed olunur, nəticə
ortalanır və YENİDƏN L2-normalize edilir (ortalama vektorun normu 1 deyil —
normalize etməsək cosine artıq cosine olmur). Buna prompt ensembling deyilir:
tək prompt CLIP mətn fəzasında səs-küylüdür, 5 şablonun ortası stil
istiqamətini sabitləşdirir.

Nəticə [len(STYLES), EMB_DIM] float32, L2-normalized — yəni `embedder`-in
şəkil embedding-ləri ilə eyni müqavilə. Skalar hasil birbaşa cosine verir.

Keş etibarlılığı:
    Fayl adı  : style_embs_{MODEL_VER}_{prompt_hash}.npz
    prompt_hash = sha256(MODEL_ID + STYLES + TEMPLATES)[:12]
    MODEL_ID, STYLES və ya TEMPLATES dəyişəndə hash dəyişir -> köhnə fayl
    sadəcə tapılmır və yenidən qurulur. Əlavə olaraq npz-in içindəki
    metadata da yoxlanılır (fayl adı toqquşmasına qarşı ikinci qapı).

İstifadə:
    python -m ml.retrieval.style_prompts              # keşi qur + 8x8 matris
    python -m ml.retrieval.style_prompts --refresh    # keşi məcburi yenidən qur
    python -m ml.retrieval.style_prompts --markdown   # README üçün cədvəl
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

if __package__ in (None, ""):  # birbaşa `python style_prompts.py` işlədiləndə
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.retrieval import config
from ml.retrieval.embedder import FashionCLIPEmbedder, l2_normalize


def build_prompts(
    styles: Sequence[str] = config.STYLES,
    templates: Sequence[str] = config.TEMPLATES,
) -> list[str]:
    """[S*T] prompt siyahısı, stil-əsas sıra ilə (s0t0, s0t1, ..., s1t0, ...).

    Sıra vacibdir — `compute_style_embeddings` nəticəni [S, T, D] formasına
    reshape edir və şablonlar üzrə ortalayır.
    """
    return [t.format(s) for s in styles for t in templates]


def prompt_hash(
    model_id: str = config.MODEL_ID,
    styles: Sequence[str] = config.STYLES,
    templates: Sequence[str] = config.TEMPLATES,
) -> str:
    """Keşi identifikasiya edən hash. Üç girişin hər hansı biri dəyişsə dəyişir.

    Model də daxildir, çünki eyni promptlar fərqli modeldə tamam başqa
    vektorlar verir — köhnə keşi yeni modellə istifadə etmək səssiz xətadır.
    """
    digest = hashlib.sha256()
    digest.update(model_id.encode("utf-8"))
    for group in (styles, templates):
        digest.update(b"\x00")
        for value in group:
            digest.update(value.encode("utf-8"))
            digest.update(b"\x1f")  # ayırıcı: ["ab","c"] != ["a","bc"]
    return digest.hexdigest()[: config.STYLE_CACHE_HASH_LEN]


def cache_path(
    model_ver: str = config.MODEL_VER,
    model_id: str = config.MODEL_ID,
    styles: Sequence[str] = config.STYLES,
    templates: Sequence[str] = config.TEMPLATES,
    cache_dir: Path | None = None,
) -> Path:
    """Bu konfiqurasiyaya uyğun keş faylının yolu."""
    directory = Path(cache_dir) if cache_dir is not None else config.STYLE_CACHE_DIR
    name = config.STYLE_CACHE_TEMPLATE.format(
        model_ver=model_ver,
        prompt_hash=prompt_hash(model_id, styles, templates),
    )
    return directory / name


@dataclass(frozen=True)
class StyleEmbeddings:
    """Stil embedding-ləri + onları doğuran konfiqurasiya.

    `vectors` : [S, EMB_DIM] float32, L2-normalized
    """

    styles: list[str]
    vectors: np.ndarray
    model_id: str
    model_ver: str
    prompt_hash: str
    path: Path
    from_cache: bool

    def similarity_matrix(self) -> np.ndarray:
        """[S, S] stil-stil cosine matrisi. Diaqonal ≈ 1.0."""
        return self.vectors @ self.vectors.T

    def index(self, style: str) -> int:
        return self.styles.index(style)

    def __len__(self) -> int:
        return len(self.styles)


def compute_style_embeddings(
    embedder: FashionCLIPEmbedder,
    styles: Sequence[str] = config.STYLES,
    templates: Sequence[str] = config.TEMPLATES,
) -> np.ndarray:
    """[S, EMB_DIM] — şablonlar üzrə ortalanmış və yenidən normalize olunmuş."""
    styles, templates = list(styles), list(templates)
    prompts = build_prompts(styles, templates)
    vectors = embedder.embed_texts(prompts)  # [S*T, D], hər sətir normalized
    per_style = vectors.reshape(len(styles), len(templates), -1)
    averaged = per_style.mean(axis=1)  # normu artıq 1 deyil ->
    return l2_normalize(averaged)  # ...ona görə yenidən normalize edilir


def _load_cache(path: Path, model_id: str, styles: list[str], templates: list[str]):
    """Keşi oxuyur. Fayl yoxdursa, xarabdırsa və ya uyğun deyilsə None."""
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            vectors = payload["vectors"].astype(np.float32)
            cached_styles = [str(s) for s in payload["styles"]]
            cached_templates = [str(t) for t in payload["templates"]]
            cached_model_id = str(payload["model_id"])
    except (OSError, ValueError, KeyError) as error:
        print(f"[xəbərdarlıq] keş oxunmadı ({type(error).__name__}: {error}) — yenidən qurulur")
        return None
    # Fayl adındakı hash uyğun gəlsə də məzmunu yoxlayırıq (paranoya deyil:
    # eyni adlı köhnə fayl əl ilə də kopyalana bilər).
    if (cached_styles, cached_templates, cached_model_id) != (styles, templates, model_id):
        print(f"[xəbərdarlıq] keş məzmunu konfiqurasiya ilə uyğun deyil — yenidən qurulur")
        return None
    if vectors.shape != (len(styles), config.EMB_DIM):
        print(f"[xəbərdarlıq] keş shape {vectors.shape} gözlənilənə uyğun deyil — yenidən qurulur")
        return None
    return vectors


def load_style_embeddings(
    embedder: FashionCLIPEmbedder | None = None,
    styles: Sequence[str] = config.STYLES,
    templates: Sequence[str] = config.TEMPLATES,
    refresh: bool = False,
    cache_dir: Path | None = None,
) -> StyleEmbeddings:
    """Keşdən oxuyur, yoxdursa hesablayıb keşləyir.

    `embedder` yalnız keş boş olanda lazımdır — verilməyibsə həmin anda
    yaradılır (model yüklənməsi ~5-10 s, keş varsa tamamilə qaçırılır).
    """
    styles, templates = list(styles), list(templates)
    path = cache_path(
        model_id=config.MODEL_ID, styles=styles, templates=templates, cache_dir=cache_dir
    )

    if not refresh:
        cached = _load_cache(path, config.MODEL_ID, styles, templates)
        if cached is not None:
            return StyleEmbeddings(
                styles=styles,
                vectors=cached,
                model_id=config.MODEL_ID,
                model_ver=config.MODEL_VER,
                prompt_hash=prompt_hash(config.MODEL_ID, styles, templates),
                path=path,
                from_cache=True,
            )

    vectors = compute_style_embeddings(embedder or FashionCLIPEmbedder(), styles, templates)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        vectors=vectors,
        styles=np.array(styles),
        templates=np.array(templates),
        model_id=np.array(config.MODEL_ID),
        model_ver=np.array(config.MODEL_VER),
    )
    return StyleEmbeddings(
        styles=styles,
        vectors=vectors,
        model_id=config.MODEL_ID,
        model_ver=config.MODEL_VER,
        prompt_hash=prompt_hash(config.MODEL_ID, styles, templates),
        path=path,
        from_cache=False,
    )


def collisions(
    styles: Sequence[str],
    matrix: np.ndarray,
    max_sim: float = config.STYLE_COLLISION_MAX,
) -> list[tuple[str, str, float]]:
    """Həddindən yaxın stil cütləri (yalnız yuxarı üçbucaq, diaqonalsız)."""
    found = []
    for i in range(len(styles)):
        for j in range(i + 1, len(styles)):
            similarity = float(matrix[i, j])
            if similarity > max_sim:
                found.append((styles[i], styles[j], similarity))
    return sorted(found, key=lambda row: -row[2])


def format_matrix(styles: Sequence[str], matrix: np.ndarray, width: int = 11) -> str:
    """8x8 matrisin terminal üçün formatı."""
    header = " " * width + "".join(f"{s[:6]:>8}" for s in styles)
    lines = [header]
    for i, style in enumerate(styles):
        row = "".join(f"{matrix[i, j]:>8.3f}" for j in range(len(styles)))
        lines.append(f"{style[:width - 1]:<{width}}{row}")
    return "\n".join(lines)


def format_markdown(styles: Sequence[str], matrix: np.ndarray) -> str:
    """README üçün markdown cədvəli."""
    head = "| | " + " | ".join(styles) + " |"
    rule = "|---|" + "---|" * len(styles)
    rows = [
        f"| **{style}** | "
        + " | ".join(f"{matrix[i, j]:.3f}" for j in range(len(styles)))
        + " |"
        for i, style in enumerate(styles)
    ]
    return "\n".join([head, rule, *rows])


def main() -> int:
    parser = argparse.ArgumentParser(description="Stil promptları və embedding keşi")
    parser.add_argument("--refresh", action="store_true", help="keşi məcburi yenidən qur")
    parser.add_argument("--markdown", action="store_true", help="matrisi markdown kimi çap et")
    parser.add_argument("--prompts", action="store_true", help="bütün promptları göstər")
    parser.add_argument(
        "--max-sim", type=float, default=config.STYLE_COLLISION_MAX,
        help="bu həddən yuxarı cütlər üçün xəbərdarlıq",
    )
    args = parser.parse_args()

    if args.prompts:
        print(f"{len(config.STYLES)} stil × {len(config.TEMPLATES)} şablon = "
              f"{len(build_prompts())} prompt")
        for prompt in build_prompts():
            print(f"  {prompt}")
        print()

    style_embs = load_style_embeddings(refresh=args.refresh)
    print(f"model      : {style_embs.model_id} ({style_embs.model_ver})")
    print(f"prompt_hash: {style_embs.prompt_hash}")
    print(f"keş        : {style_embs.path}")
    print(f"mənbə      : {'KEŞ' if style_embs.from_cache else 'YENİDƏN HESABLANDI'}")
    norms = np.linalg.norm(style_embs.vectors, axis=1)
    print(f"vektorlar  : shape={style_embs.vectors.shape} dtype={style_embs.vectors.dtype} "
          f"norm=[{norms.min():.4f}, {norms.max():.4f}]")

    matrix = style_embs.similarity_matrix()
    print(f"\n--- {len(style_embs)}x{len(style_embs)} stil-stil oxşarlıq matrisi ---")
    print(format_markdown(style_embs.styles, matrix) if args.markdown
          else format_matrix(style_embs.styles, matrix))

    off_diagonal = matrix[~np.eye(len(style_embs), dtype=bool)]
    print(f"\ndiaqonaldan kənar: orta={off_diagonal.mean():.3f} "
          f"max={off_diagonal.max():.3f} min={off_diagonal.min():.3f}")

    clashing = collisions(style_embs.styles, matrix, args.max_sim)
    if clashing:
        print(f"\n[XƏBƏRDARLIQ] {len(clashing)} cüt {args.max_sim}-dan yuxarıdır — "
              "bu stillər praktikada fərqlənmir:")
        for left, right, similarity in clashing:
            print(f"  {left} ↔ {right}: {similarity:.3f}")
        print(f"  Ehtiyat namizədlər: {', '.join(config.STYLE_FALLBACK_CANDIDATES)}")
        return 1
    print(f"\n[OK] heç bir cüt {args.max_sim}-dan yuxarı deyil — 8 stil ayırd edilə bilir.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
