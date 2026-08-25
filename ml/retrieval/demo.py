"""Demo — sorğu + top-5 nəticəni PNG grid kimi yazır və latency ölçür.

Çıxış: `ml/retrieval/outputs/demo_*.png`

Latency iki hissəyə ayrılır:
    embed  — sorğunun vektora çevrilməsi (model inference, CPU)
    search — vector store-da axtarış
Model yüklənməsi (soyuq başlanğıc) ölçüyə DAXİL EDİLMİR: real xidmətdə
model bir dəfə yüklənir və prosesdə qalır. Ayrıca göstərilir.

İstifadə:
    python -m ml.retrieval.demo --text "black leather boots"
    python -m ml.retrieval.demo --image data/images/item_0002.jpg --exclude-self
    python -m ml.retrieval.demo --all          # bir neçə nümunə sorğu
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # başsız (headless) mühit — ekran tələb etmir

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.retrieval import config
from ml.retrieval.search import Searcher
from ml.retrieval.store.base import SearchResult

DEMO_QUERIES = [
    ("text", "black leather boots"),
    ("text", "a floral summer dress"),
    ("text", "blue denim jeans"),
    ("image", "item_0003"),
]


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:48]


def _thumbnail(path: str | Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(config.DEMO_THUMB_SIZE, Image.LANCZOS)
    return image


def _draw_placeholder(axis, message: str) -> None:
    axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=9, wrap=True)
    axis.set_facecolor("#f0f0f0")


def render_grid(
    results: list[SearchResult],
    output_path: Path,
    query_text: str | None = None,
    query_image: str | Path | None = None,
    timing: dict[str, float] | None = None,
    backend: str = config.BACKEND,
) -> Path:
    """Sorğu + nəticələri bir sətirdə PNG kimi yazır."""
    columns = 1 + len(results)
    figure, axes = plt.subplots(
        1, columns, figsize=(2.2 * columns, 3.6), dpi=config.DEMO_FIG_DPI
    )
    axes = np.atleast_1d(axes)

    # --- sorğu xanası ---
    axis = axes[0]
    if query_image is not None:
        axis.imshow(_thumbnail(query_image))
        axis.set_title(f"SORĞU (şəkil)\n{Path(query_image).name}", fontsize=8, color="#0b5")
    else:
        _draw_placeholder(axis, f'"{query_text}"')
        axis.set_title("SORĞU (mətn)", fontsize=8, color="#0b5")
    axis.set_xticks([]); axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_edgecolor("#0b5"); spine.set_linewidth(2)

    # --- nəticələr ---
    for position, result in enumerate(results, start=1):
        axis = axes[position]
        image_path = result.meta.get("image_path")
        if image_path and Path(image_path).exists():
            axis.imshow(_thumbnail(image_path))
        else:
            _draw_placeholder(axis, "(şəkil yoxdur)")
        label = (result.meta.get("text") or result.meta.get("category") or "")[:34]
        axis.set_title(
            f"#{position}  cos={result.score:.3f}\n{result.item_id}\n{label}", fontsize=7
        )
        axis.set_xticks([]); axis.set_yticks([])

    subtitle = f"backend={backend}"
    if timing:
        target = config.LATENCY_TARGET_SEC
        ok = "✓" if timing["total_sec"] < target else "✗"
        subtitle += (
            f"   embed={timing['embed_sec'] * 1000:.0f} ms + "
            f"search={timing['search_sec'] * 1000:.2f} ms = "
            f"{timing['total_sec'] * 1000:.0f} ms  "
            f"(hədəf < {target:.0f} s {ok})"
        )
    figure.suptitle(subtitle, fontsize=9, y=0.04)
    figure.tight_layout(rect=(0, 0.06, 1, 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def measure_latency(searcher: Searcher, runs: int = config.DEMO_LATENCY_RUNS, **query) -> dict:
    """Sorğunu `runs` dəfə işlədib median latency qaytarır."""
    embeds, searches = [], []
    results = []
    for _ in range(runs):
        results, timing = searcher.search_timed(**query)
        embeds.append(timing["embed_sec"])
        searches.append(timing["search_sec"])
    return {
        "results": results,
        "embed_sec": float(np.median(embeds)),
        "search_sec": float(np.median(searches)),
        "total_sec": float(np.median(embeds)) + float(np.median(searches)),
        "runs": runs,
    }


def run_demo(
    text: str | None = None,
    image: str | Path | None = None,
    k: int = config.DEFAULT_TOP_K,
    backend: str | None = None,
    searcher: Searcher | None = None,
    exclude_self: bool = False,
    output_dir: Path = config.OUTPUT_DIR,
) -> dict:
    owns_searcher = searcher is None
    searcher = searcher or Searcher(backend=backend)
    try:
        exclude = [Path(image).stem] if (image and exclude_self) else None
        measurement = measure_latency(
            searcher, text=text, image=image, k=k, exclude=exclude
        )
        name = _slugify(text if text else Path(image).stem)
        kind = "text" if text else "image"
        output_path = Path(output_dir) / f"demo_{kind}_{name}.png"
        render_grid(
            measurement["results"],
            output_path,
            query_text=text,
            query_image=image,
            timing=measurement,
            backend=backend or config.BACKEND,
        )
        return {**measurement, "path": output_path, "query": text or str(image)}
    finally:
        if owns_searcher:
            searcher.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo: top-5 nəticə -> PNG + latency")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text")
    group.add_argument("--image")
    group.add_argument("--all", action="store_true", help="bir neçə nümunə sorğu işlət")
    parser.add_argument("-k", "--top-k", type=int, default=config.DEFAULT_TOP_K)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--exclude-self", action="store_true")
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR)
    args = parser.parse_args()

    cold_start = time.perf_counter()
    searcher = Searcher(backend=args.backend)
    searcher.encode_text("warmup")  # modeli yüklə
    cold_sec = time.perf_counter() - cold_start

    if searcher.store.count() == 0:
        print("Store boşdur -> python -m ml.retrieval.ingest")
        return 1

    print(f"backend        : {args.backend or config.BACKEND} "
          f"({searcher.store.count()} əşya)")
    print(f"soyuq başlanğıc: {cold_sec:.2f} san (model yüklənməsi — bir dəfəlik)")
    print(f"latency medianı: {config.DEMO_LATENCY_RUNS} təkrar\n")

    queries = (
        DEMO_QUERIES if args.all
        else [("text", args.text)] if args.text
        else [("image", args.image)]
    )

    rows = []
    for kind, value in queries:
        if kind == "image":
            path = value if Path(value).exists() else \
                Path(config.IMAGE_DIR) / f"{value}{config.SAMPLE_IMAGE_EXT}"
            result = run_demo(image=path, k=args.top_k, backend=args.backend,
                              searcher=searcher, exclude_self=True,
                              output_dir=Path(args.output_dir))
        else:
            result = run_demo(text=value, k=args.top_k, backend=args.backend,
                              searcher=searcher, exclude_self=args.exclude_self,
                              output_dir=Path(args.output_dir))
        rows.append((kind, result))
        print(f"[{kind}] {result['query']}")
        print(f"    top-{args.top_k}: " + ", ".join(
            f"{r.item_id}({r.score:.3f})" for r in result["results"]))
        print(f"    latency: embed {result['embed_sec'] * 1000:6.1f} ms + "
              f"search {result['search_sec'] * 1000:5.2f} ms = "
              f"{result['total_sec'] * 1000:6.1f} ms")
        print(f"    PNG    : {result['path']}")

    searcher.close()

    total = [r["total_sec"] for _, r in rows]
    worst = max(total)
    target = config.LATENCY_TARGET_SEC
    print(f"\n--- latency xülasəsi ({len(rows)} sorğu) ---")
    print(f"ən pis top-{args.top_k}: {worst * 1000:.1f} ms | hədəf < {target * 1000:.0f} ms "
          f"-> {'✓ ÖDƏNİLİR' if worst < target else '✗ ÖDƏNMİR'}")
    print(f"ehtiyat        : {target / worst:.0f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
