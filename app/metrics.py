from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable

SENSEVOICE_TAG_RE = re.compile(r"<\|[^|]+?\|>")
FILLER_WORDS = ("呃", "嗯", "唔", "呐")


def normalize_text(text: str) -> str:
    text = SENSEVOICE_TAG_RE.sub("", text)
    text = unicodedata.normalize("NFKC", text).lower()
    chars = []
    for char in text:
        category = unicodedata.category(char)
        if char.isspace() or category[0] in {"P", "S"}:
            continue
        chars.append(char)
    return "".join(chars)


def normalize_text_clean(text: str) -> str:
    normalized = normalize_text(text)
    for word in FILLER_WORDS:
        normalized = normalized.replace(word, "")
    return normalized


def edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(
    expected: str,
    actual: str,
    *,
    normalizer: Callable[[str], str] = normalize_text,
) -> tuple[float, int, int]:
    normalized_expected = normalizer(expected)
    normalized_actual = normalizer(actual)
    distance = edit_distance(normalized_expected, normalized_actual)
    expected_chars = max(1, len(normalized_expected))
    return distance / expected_chars, distance, expected_chars


def contains_reference(expected: str, actual: str) -> bool:
    normalized_expected = normalize_text_clean(expected)
    normalized_actual = normalize_text_clean(actual)
    return bool(normalized_expected) and normalized_expected in normalized_actual
