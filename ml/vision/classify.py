from dataclasses import dataclass


@dataclass
class ClassifiedItem:
    item_id: str
    true_category: str
    predicted_category: str
    confidence: float


def filter_by_confidence(
    items: list[ClassifiedItem],
    threshold: float,
) -> list[ClassifiedItem]:
    return [item for item in items if item.confidence >= threshold]
