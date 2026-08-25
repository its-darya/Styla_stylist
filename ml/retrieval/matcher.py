"""Reference outfit matcher.

Referens outfit-in əşyalarını istifadəçinin qarderobu ilə tutuşdurur:
    - hər əşya üçün qarderobda ən yaxın namizədləri tapır
    - ən yaxşı oxşarlıq `config.MATCH_THRESHOLD`-dan aşağıdırsa əşyanı
      "missing" işarələyir (E modulu bunlar üçün kataloqdan alternativ təklif edir)
    - kateqoriya filtri: zero-shot CLIP mətn promptları ilə əşyanın kobud
      kateqoriyası təyin olunur, axtarış həmin kateqoriya ilə məhdudlaşır

Kateqoriya filtri necə işləyir:
    1. Referens əşyanın şəkli `config.CATEGORIES` promptları ilə təsnif olunur
       ("a photo of a {}, a type of clothing") -> kobud etiket, məs. "boots".
    2. Qarderobdakı mövcud kateqoriya adları (məs. "Ankle Booties", "Boots")
       MƏTN oxşarlığı ilə həmin kobud etiketlərə xəritələnir.
    3. Axtarış `where={"category": [uyğun gələn adlar]}` ilə edilir.
    Bu yanaşma hər iki backend-də işləyir, çünki yalnız `category` sütununa
    söykənir (pgvector sxemində də var).

İstifadə:
    python -m ml.retrieval.matcher --reference data/images/item_0002.jpg
    python -m ml.retrieval.matcher --reference-outfit 100002074
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.retrieval import config
from ml.retrieval.search import Searcher
from ml.retrieval.store.base import SearchResult

STATUS_MATCHED = "matched"
STATUS_MISSING = "missing"


@dataclass
class ItemMatch:
    """Referens outfit-in bir əşyası üçün nəticə."""

    query: str                       # referens şəklin yolu / id-si
    status: str                      # matched | missing
    score: float                     # ən yaxşı oxşarlıq
    threshold: float
    predicted_category: str | None = None
    category_confidence: float = 0.0
    category_filter: list[str] = field(default_factory=list)
    best_match_id: str | None = None
    best_match_meta: dict[str, Any] = field(default_factory=dict)
    candidates: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_missing(self) -> bool:
        return self.status == STATUS_MISSING


@dataclass
class OutfitMatchReport:
    items: list[ItemMatch]
    threshold: float

    @property
    def matched(self) -> list[ItemMatch]:
        return [i for i in self.items if not i.is_missing]

    @property
    def missing(self) -> list[ItemMatch]:
        return [i for i in self.items if i.is_missing]

    @property
    def coverage(self) -> float:
        """Referens outfit-in neçə faizi qarderobla təkrarlana bilir."""
        return len(self.matched) / len(self.items) if self.items else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "coverage": self.coverage,
            "matched_count": len(self.matched),
            "missing_count": len(self.missing),
            "items": [asdict(i) for i in self.items],
        }


class CategoryClassifier:
    """Zero-shot kateqoriya təsnifatı — CLIP mətn promptları ilə."""

    def __init__(self, embedder, categories: Sequence[str] = config.CATEGORIES) -> None:
        self.categories = list(categories)
        prompts = [config.CATEGORY_PROMPT_TEMPLATE.format(c) for c in self.categories]
        self._prompt_vectors = embedder.embed_texts(prompts)
        self._embedder = embedder

    def classify_vector(self, vector: np.ndarray) -> tuple[str, float]:
        """Embedding -> (kobud kateqoriya, softmax ehtimalı).

        Standart zero-shot CLIP: cosine balları `logit_scale` ilə miqyaslanır
        və softmax tətbiq olunur. Xam cosine fərqləri (~0.01–0.05) birbaşa
        inam ölçüsü kimi yararsızdır — miqyassız hər şey "aşağı inam" görünür.
        """
        scores = self._prompt_vectors @ np.asarray(vector, dtype=np.float32).reshape(-1)
        logits = scores.astype(np.float64) * config.CATEGORY_LOGIT_SCALE
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        best = int(np.argmax(probabilities))
        return self.categories[best], float(probabilities[best])

class ColorClassifier:
    """Zero-shot rəng təsnifatı — CLIP mətn promptları ilə."""

    COLORS = [
        "black", "white", "gray", "red", "blue", "green", "yellow", 
        "orange", "purple", "pink", "brown", "beige", "navy", "maroon",
        "black & white", "navy & white", "blue & white", "red & white",
        "multi-color"
    ]

    def __init__(self, embedder) -> None:
        self.colors = self.COLORS
        # Prompt for color
        prompts = [f"a photo of a {c} clothing item" for c in self.colors]
        self._prompt_vectors = embedder.embed_texts(prompts)
        self._embedder = embedder

    def classify_vector(self, vector: np.ndarray) -> tuple[str, float]:
        scores = self._prompt_vectors @ np.asarray(vector, dtype=np.float32).reshape(-1)
        logits = scores.astype(np.float64) * config.CATEGORY_LOGIT_SCALE
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        best = int(np.argmax(probabilities))
        return self.colors[best], float(probabilities[best])

class PatternClassifier:
    """Zero-shot naxış (pattern) təsnifatı — CLIP mətn promptları ilə."""

    PATTERNS = ["Solid", "Ribbed", "Pinstripe", "Check", "Textured", "Floral", "Polka Dot", "Striped"]

    def __init__(self, embedder) -> None:
        self.patterns = self.PATTERNS
        prompts = [f"a photo of a {p.lower()} clothing item" for p in self.patterns]
        self._prompt_vectors = embedder.embed_texts(prompts)
        self._embedder = embedder

    def classify_vector(self, vector: np.ndarray) -> tuple[str, float]:
        scores = self._prompt_vectors @ np.asarray(vector, dtype=np.float32).reshape(-1)
        logits = scores.astype(np.float64) * config.CATEGORY_LOGIT_SCALE
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        best = int(np.argmax(probabilities))
        return self.patterns[best], float(probabilities[best])

    def map_labels(self, labels: Sequence[str]) -> dict[str, str]:
        """Qarderob kateqoriya adlarını kobud kateqoriyalara xəritələyir.

        Mətn-mətn oxşarlığı: "Ankle Booties" -> "boots".
        Oxşarlıq `config.CATEGORY_MAP_MIN_SIM`-dən aşağıdırsa xəritələnmir.
        """
        labels = [l for l in dict.fromkeys(labels) if l]
        if not labels:
            return {}
        label_vectors = self._embedder.embed_texts(list(labels))
        similarity = label_vectors @ self._prompt_vectors.T
        mapping: dict[str, str] = {}
        for i, label in enumerate(labels):
            best = int(np.argmax(similarity[i]))
            if float(similarity[i][best]) >= config.CATEGORY_MAP_MIN_SIM:
                mapping[label] = self.categories[best]
        return mapping


class Matcher:
    def __init__(
        self,
        searcher: Searcher | None = None,
        backend: str | None = None,
        threshold: float = config.MATCH_THRESHOLD,
        use_category_filter: bool = config.CATEGORY_FILTER_ENABLED,
    ) -> None:
        self.searcher = searcher or Searcher(backend=backend)
        self.threshold = threshold
        self.use_category_filter = use_category_filter
        self.classifier = CategoryClassifier(self.searcher.embedder)
        self._label_map: dict[str, str] | None = None

    # --- kateqoriya filtri -----------------------------------------------
    def _wardrobe_labels(self) -> list[str]:
        """Qarderobda mövcud kateqoriya adları — hər iki backend-də işləyir."""
        return self.searcher.store.distinct_values("category")

    def _category_filter_for(self, coarse: str) -> list[str]:
        """Kobud etiketə uyğun gələn qarderob kateqoriya adları."""
        if self._label_map is None:
            self._label_map = self.classifier.map_labels(self._wardrobe_labels())
        return sorted({label for label, c in self._label_map.items() if c == coarse})

    # --- əsas ------------------------------------------------------------
    def match_item(
        self,
        image: str | Path | None = None,
        vector: np.ndarray | None = None,
        k: int = config.DEFAULT_TOP_K,
        exclude: Sequence[str] | None = None,
    ) -> ItemMatch:
        """Bir referens əşyasını qarderobla tutuşdurur."""
        if vector is None:
            if image is None:
                raise ValueError("image və ya vector lazımdır")
            vector = self.searcher.encode_image(image)
        query_label = str(image) if image is not None else "<vector>"

        predicted, confidence = self.classifier.classify_vector(vector)
        category_filter: list[str] = []
        where = None
        if self.use_category_filter and confidence >= config.CATEGORY_CONFIDENCE_MIN:
            category_filter = self._category_filter_for(predicted)
            if category_filter:
                where = {"category": category_filter}

        results = self.searcher.search(vector=vector, k=k, where=where, exclude=exclude)
        if where is not None and (not results or results[0].score < self.threshold):
            # Filtr ya hər şeyi kəsib, ya da yalnız zəif namizədlər qalıb.
            # Zero-shot təsnifat səhv ola bilər (məs. palto -> "boots"), ona görə
            # "missing" qərarını filtrin xətasına əsaslandırmırıq: filtrsiz
            # təkrar axtarırıq və daha yaxşı nəticəni götürürük.
            fallback = self.searcher.search(vector=vector, k=k, exclude=exclude)
            if fallback and (not results or fallback[0].score > results[0].score):
                category_filter = []
                results = fallback

        best = results[0] if results else None
        score = float(best.score) if best else 0.0
        return ItemMatch(
            query=query_label,
            status=STATUS_MATCHED if score >= self.threshold else STATUS_MISSING,
            score=score,
            threshold=self.threshold,
            predicted_category=predicted,
            category_confidence=confidence,
            category_filter=category_filter,
            best_match_id=best.item_id if best else None,
            best_match_meta=dict(best.meta) if best else {},
            candidates=[
                {"item_id": r.item_id, "score": float(r.score),
                 "category": r.meta.get("category"), "text": r.meta.get("text")}
                for r in results
            ],
        )

    def match_outfit(
        self,
        images: Sequence[str | Path],
        k: int = config.DEFAULT_TOP_K,
        exclude: Sequence[str] | None = None,
    ) -> OutfitMatchReport:
        """Referens outfit-in bütün əşyalarını qarderobla tutuşdurur."""
        items = [self.match_item(image=p, k=k, exclude=exclude) for p in images]
        return OutfitMatchReport(items=items, threshold=self.threshold)

    def close(self) -> None:
        self.searcher.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def format_report(report: OutfitMatchReport) -> str:
    lines = [
        f"Threshold : {report.threshold}",
        f"Coverage  : {report.coverage:.0%} "
        f"({len(report.matched)} tapıldı / {len(report.missing)} əskik)",
        "",
    ]
    for item in report.items:
        mark = "✓" if not item.is_missing else "✗"
        lines.append(f"{mark} {Path(item.query).name}")
        lines.append(
            f"    kateqoriya : {item.predicted_category} "
            f"(inam {item.category_confidence:.2f})"
            + (f" | filtr: {item.category_filter}" if item.category_filter else " | filtr yoxdur")
        )
        if item.best_match_id:
            meta = item.best_match_meta
            lines.append(
                f"    ən yaxın   : {item.best_match_id} cos={item.score:.4f} "
                f"[{meta.get('category')}] {meta.get('text') or ''}"
            )
        else:
            lines.append("    ən yaxın   : (namizəd yoxdur)")
        lines.append(f"    status     : {item.status.upper()}")
    return "\n".join(lines)


def _outfit_items(outfit_id: str, image_dir: Path = config.IMAGE_DIR) -> list[Path]:
    meta_path = Path(image_dir) / Path(config.IMAGE_META_PATH).name
    items = json.loads(meta_path.read_text(encoding="utf-8"))["items"]
    return [
        Path(image_dir) / info["file"]
        for info in items.values()
        if info.get("outfit_id") == outfit_id
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Reference outfit matcher")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--reference", action="append", help="referens şəkil yolu (təkrarlana bilər)")
    group.add_argument("--reference-outfit", help="meta.json-dakı outfit_id")
    parser.add_argument("-k", "--top-k", type=int, default=config.DEFAULT_TOP_K)
    parser.add_argument("--threshold", type=float, default=config.MATCH_THRESHOLD)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--no-category-filter", action="store_true")
    parser.add_argument("--exclude-self", action="store_true",
                        help="referens əşyanın öz id-sini qarderobdan çıxar (dürüst test)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    images = (
        [Path(p) for p in args.reference]
        if args.reference else _outfit_items(args.reference_outfit)
    )
    if not images:
        print("Referens şəkil tapılmadı")
        return 1
    exclude = [p.stem for p in images] if args.exclude_self else None

    with Matcher(
        backend=args.backend,
        threshold=args.threshold,
        use_category_filter=not args.no_category_filter,
    ) as matcher:
        if matcher.searcher.store.count() == 0:
            print("Store boşdur -> python -m ml.retrieval.ingest")
            return 1
        report = matcher.match_outfit(images, k=args.top_k, exclude=exclude)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
