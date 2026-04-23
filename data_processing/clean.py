"""Text normalization and quality filters for aviation safety narratives."""
from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")
_NONPRINT = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_PAGE_ARTIFACT = re.compile(
    r"NTSB/[A-Z]+-\d{2}/\d{2}|"
    r"\bPage\s+\d+(\s+of\s+\d+)?\b|"
    r"\bWashington,?\s+D\.?C\.?\s+20594\b",
    re.IGNORECASE,
)
_HEADER_FRAGMENTS = re.compile(
    r"National Transportation Safety Board|"
    r"E\s*PLURIBUS\s*U\s*NUM|"
    r"SAFETYBO\s*AR\s*D",
    re.IGNORECASE,
)

# Causal / reasoning discourse markers. Requires at least one to confirm the
# narrative carries a reason-effect link rather than being a bare log line.
_CAUSAL_TERMS = [
    r"because", r"due to", r"caused by", r"causing", r"resulted in", r"resulting in",
    r"led to", r"leading to", r"as a result", r"attributed to", r"contributed to",
    r"contributing (factor|cause)", r"probable cause", r"root cause", r"factor(?:s)? in",
    r"failed to", r"failure to", r"did not", r"was unable to", r"attempted to",
    r"after (?:the|a|an)?", r"when the", r"while (?:the|taxi|climb|cruis|landing|approach|descend)",
    r"during (?:the|takeoff|taxi|climb|cruis|landing|approach|descent|rollout|rotation)",
    r"prior to", r"subsequently", r"thereafter", r"consequently", r"therefore",
    r"in order to", r"so that", r"such that", r"triggered", r"induced",
    r"loss of control", r"lost control", r"encountered", r"experienced",
]
_CAUSAL_RE = re.compile(r"\b(?:" + "|".join(_CAUSAL_TERMS) + r")\b", re.IGNORECASE)


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = _NONPRINT.sub(" ", t)
    t = t.replace("\u00ad", "")  # soft hyphen
    t = re.sub(r"-\n(?=\w)", "", t)  # de-hyphenate line breaks
    t = t.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    t = _HEADER_FRAGMENTS.sub(" ", t)
    t = _PAGE_ARTIFACT.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    return t


def word_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def has_causal_link(text: str) -> bool:
    if not text:
        return False
    return bool(_CAUSAL_RE.search(text))
