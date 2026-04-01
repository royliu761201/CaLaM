from __future__ import annotations

from typing import Any

PROMPT_TEXT_KEYS = (
    "prompt",
    "question",
    "instruction",
    "goal",
    "query",
    "text",
    "content",
)
NESTED_PROMPT_TEXT_KEYS = ("text", "prompt", "content")

def _coerce_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        for key in NESTED_PROMPT_TEXT_KEYS:
            nested = value.get(key)
            if isinstance(nested, str):
                return nested

    if isinstance(value, (list, tuple)):
        parts = []
        for item in value:
            text = _coerce_text(item)
            if text:
                parts.append(text)
        if parts:
            return " ".join(parts)

    return None

def extract_prompt_text(sample: Any, *, strict: bool = False) -> str:
    if not isinstance(sample, dict):
        return str(sample)

    for key in PROMPT_TEXT_KEYS:
        if key not in sample:
            continue
        text = _coerce_text(sample[key])
        if text is not None:
            return text

    if strict:
        raise ValueError(
            f"Unable to extract prompt text from sample keys: {sorted(sample.keys())}"
        )
    return ""

def prompt_length(sample: Any) -> int:
    return len(extract_prompt_text(sample, strict=False))

def select_longest_samples(samples: list[Any], count: int) -> list[Any]:
    ordered = list(samples)
    ordered.sort(key=prompt_length, reverse=True)
    return ordered[:count]

def sort_samples_by_prompt_length(samples: list[Any], *, reverse: bool = False) -> list[Any]:
    ordered = list(samples)
    ordered.sort(key=prompt_length, reverse=reverse)
    return ordered
