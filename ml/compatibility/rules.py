from ml.vision.color import ColorSpec

SOLID_PATTERN_ALIASES = {"solid", "plain", "none", "", "unpatterned"}

DEFAULT_NEUTRAL_SATURATION = 0.15
DEFAULT_CLASH_LOW = 45.0
DEFAULT_CLASH_HIGH = 135.0


def hue_distance(h1: float, h2: float) -> float:
    d = abs(h1 - h2) % 360.0
    return min(d, 360.0 - d)


def is_neutral(spec: ColorSpec, sat_threshold: float = DEFAULT_NEUTRAL_SATURATION) -> bool:
    return spec.saturation < sat_threshold


def color_clash(
    a: ColorSpec,
    b: ColorSpec,
    *,
    sat_threshold: float = DEFAULT_NEUTRAL_SATURATION,
    low: float = DEFAULT_CLASH_LOW,
    high: float = DEFAULT_CLASH_HIGH,
) -> bool:
    if is_neutral(a, sat_threshold) or is_neutral(b, sat_threshold):
        return False
    d = hue_distance(a.hue, b.hue)
    return low < d < high


def pattern_clash(patterns: list[str]) -> bool:
    strong = [p for p in patterns if p.strip().lower() not in SOLID_PATTERN_ALIASES]
    return len(strong) >= 2
