"""Text normalization and quality filters for aviation safety narratives."""
from __future__ import annotations

import re
import unicodedata
from typing import List

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


# --------------------------------------------------------------------------- #
# Cross-source blocking helpers
# --------------------------------------------------------------------------- #

# ICAO-style registration prefixes that actually appear in the corpus.
# Narrow list on purpose: a broad "any letter + dash + alnum" catches too
# many incidental tokens (e.g. model designators like "DHC-8", runway
# labels like "08-26", equipment IDs like "PT6A-34AG"). The ones listed
# cover the countries whose agencies we ingest (US/CA/FR/GB/DE/AU/ES/IT/JP)
# plus a few common operators of US-managed fleets.
_TAIL_PATTERNS = [
    # N-numbers: N + 1-5 digits + 0-2 trailing letters. Disallow hyphen.
    re.compile(r"\bN\d{1,5}[A-Z]{0,2}\b"),
    # Canadian (C-FXXX / C-GXXX / CF-XXX), French (F-XXXX), German (D-XXXX),
    # British (G-XXXX), Spanish (EC-XXX), Italian (I-XXXX), Australian (VH-XXX),
    # Japanese (JA + 4), Irish (EI-XXX).
    re.compile(r"\b(?:C-[FG][A-Z]{3}|CF-[A-Z]{3,4})\b"),
    re.compile(r"\b[FDGI]-[A-Z]{4}\b"),
    re.compile(r"\b(?:EC|EI|VH|VT)-[A-Z]{3}\b"),
    re.compile(r"\bJA\d{3,4}[A-Z]?\b"),
]

_ISO_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")
_DMY_DATE = re.compile(r"^(\d{2})/(\d{2})/(\d{4})")
_YMD_COMPACT = re.compile(r"^(\d{4})(\d{2})(\d{2})$")
_YYMMDD_SLASH = re.compile(r"^(\d{2})/(\d{2})/(\d{2})")  # legacy NTSB


def normalize_date(raw: str) -> str:
    """Return YYYYMMDD or '' if unparseable.

    Handles BEA (DD/MM/YYYY), NTSB_REPORT (YYYY-MM-DD), FAA_AIDS
    (YYYYMMDD in ``c9``), TSB (``YYYY-MM-DD HH:MM:SS.fffffff``), and the
    PRE1982 ``MM/DD/YY HH:MM:SS`` format. Anything else is dropped.
    """
    if not raw:
        return ""
    s = str(raw).strip()
    m = _ISO_DATE.match(s)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}"
    m = _YMD_COMPACT.match(s)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}"
    m = _DMY_DATE.match(s)
    if m:
        # Ambiguous — BEA is DD/MM/YYYY, legacy US AIDS would be MM/DD/YYYY.
        # In this corpus only BEA reaches this branch (FAA AIDS c9 is
        # already YYYYMMDD), so treat as DD/MM/YYYY.
        return f"{m.group(3)}{m.group(2)}{m.group(1)}"
    m = _YYMMDD_SLASH.match(s)
    if m:
        yy = int(m.group(3))
        year = 1900 + yy if yy >= 50 else 2000 + yy
        return f"{year}{m.group(1)}{m.group(2)}"
    return ""


def extract_tails(text: str) -> List[str]:
    """Return all registration-looking tokens in ``text``.

    Used to recover a tail number when the source has no structured
    registration column (TSB_CANADA, NTSB_REPORT text, NTSB_ASRS narratives).
    """
    if not text:
        return []
    found: list[str] = []
    for pat in _TAIL_PATTERNS:
        found.extend(pat.findall(text))
    # Preserve first-seen order, dedupe case-insensitively.
    seen: set[str] = set()
    uniq: list[str] = []
    for t in found:
        k = t.upper()
        if k not in seen:
            seen.add(k)
            uniq.append(k)
    return uniq


def normalize_tail(raw: str) -> str:
    """Canonicalize a tail: uppercase, strip whitespace, strip leading 'N'-zero.

    Some sources write ``N0123A`` vs ``N123A``; the FAA registry itself
    allows both. We canonicalize by uppercasing and removing the leading
    zero after 'N' (US-specific).
    """
    if not raw:
        return ""
    t = str(raw).strip().upper().replace(" ", "")
    if t.startswith("N0") and len(t) > 2 and t[2].isdigit():
        t = "N" + t[2:]
    return t

