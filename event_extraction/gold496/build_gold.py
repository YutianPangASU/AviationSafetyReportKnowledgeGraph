"""Build the gold evaluation set from the full NTSB investigation reports.

Each usable report (AAR / AIR / AAB with extracted text) is split into:
  - INPUT: the report text BEFORE the Conclusions / Probable Cause section.
    This is what extraction models see. The investigator-written conclusions
    are withheld so the model cannot read the answer.
  - GOLD: the human annotations:
      * findings_items      - numbered Findings list (AAR/AIR)
      * probable_cause_text - the Probable Cause statement
      * coded_*             - NTSB database supervision joined via ntsb_no
                              (occurrences, seq_of_events, coded findings)

Outputs:
  gold.jsonl                 one record per report (gold + metadata)
  inputs/<report_number>.txt the model-visible narrative input
  build_report.json          parse coverage summary

Usage: python3 event_extraction/gold496/build_gold.py
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "data/NTSB_REPORTS/manifest.csv"
ENRICHED = REPO / "data/corpus/corpus_enriched.jsonl"
OUT_DIR = Path(__file__).resolve().parent
INPUTS_DIR = OUT_DIR / "inputs"
MAX_INPUT_CHARS = 700_000  # safety cap; keeps the longest report inside model context
MIN_INPUT_CHARS = 5_000  # below this the "report" is a cover page or stub, not a narrative

TYPES = {"AAR", "AIR", "AAB"}


def find_heading(text: str, pattern: str, min_frac: float = 0.30) -> list[int]:
    """Return offsets of lines matching `pattern` past `min_frac` of the doc."""
    hits = []
    floor = int(len(text) * min_frac)
    for m in re.finditer(pattern, text, re.MULTILINE):
        if m.start() >= floor:
            hits.append(m.start())
    return hits


def split_aar(text: str):
    """AAR / AIR: locate Conclusions -> Findings -> Probable Cause."""
    # Conclusions heading: standalone-ish line, possibly "3. Conclusions".
    concl_pat = r"^\s{0,8}(?:\d+\.\s*)?(?:C\s?O\s?N\s?C\s?L\s?U\s?S\s?I\s?O\s?N\s?S?|Conclusions?|CONCLUSIONS?)\s*$"
    find_pat = r"^\s{0,8}(?:\d+\.\d*\s*)?(?:Findings|FINDINGS)\s*$"
    pc_pat = r"^\s{0,8}(?:\d+\.\d*\s*)?(?:(?:The\s+)?Probable\s+Cause|PROBABLE\s+CAUSE)\b.{0,40}$"

    concl_hits = find_heading(text, concl_pat)
    split_at = None
    findings_at = None
    for off in concl_hits:
        # real Conclusions section is followed by a Findings subsection nearby
        m = re.search(find_pat, text[off : off + 4000], re.MULTILINE)
        if m:
            split_at = off
            findings_at = off + m.start()
    if split_at is None and concl_hits:
        split_at = concl_hits[-1]

    pc_hits = find_heading(text, pc_pat)
    pc_at = pc_hits[-1] if pc_hits else None
    if split_at is None and pc_at is not None:
        split_at = pc_at
    if split_at is None:
        return None

    findings_items = []
    if findings_at is not None:
        end = pc_at if (pc_at and pc_at > findings_at) else min(len(text), findings_at + 30_000)
        block = text[findings_at:end]
        findings_items = parse_numbered_items(block)

    pc_text = ""
    if pc_at is not None:
        tail = text[pc_at : pc_at + 8000]
        # cut at the next major heading (Recommendations / Appendix / numbered top section)
        m = re.search(
            r"^\s{0,8}(?:\d+\.\s*)?(?:Recommendations|RECOMMENDATIONS|Appendix|APPENDIX|Safety\s+Recommendations?)\b",
            tail[80:],
            re.MULTILINE,
        )
        pc_text = tail[: 80 + m.start()] if m else tail[:4000]
        pc_text = clean_section(pc_text)

    # Fallback gold: the raw conclusions section text, for reports whose
    # findings are unnumbered prose (common in modern AIR reports).
    concl_text = ""
    if not findings_items and not pc_text:
        concl_text = clean_section(text[split_at : split_at + 10_000])[:8000]

    return {
        "split_at": split_at,
        "findings_items": findings_items,
        "probable_cause_text": pc_text,
        "conclusions_text": concl_text,
    }


def split_aab(text: str):
    """AAB briefs: PROBABLE CAUSE heading only."""
    pc_pat = r"^\s{0,8}(?:(?:The\s+)?Probable\s+Cause|PROBABLE\s+CAUSE)\s*$"
    hits = find_heading(text, pc_pat, min_frac=0.20)
    if not hits:
        return None
    pc_at = hits[-1]
    pc_text = clean_section(text[pc_at : pc_at + 6000])
    return {
        "split_at": pc_at,
        "findings_items": [],
        "probable_cause_text": pc_text,
        "conclusions_text": "",
    }


def parse_numbered_items(block: str) -> list[str]:
    """Split a Findings block into its numbered items."""
    parts = re.split(r"\n\s{0,8}(\d{1,3})\.\s+", block)
    items = []
    # parts: [pre, num, body, num, body, ...]
    for i in range(1, len(parts) - 1, 2):
        body = parts[i + 1]
        # stop an item at a blank-line-then-heading boundary
        body = re.split(r"\n\s*\n\s*[A-Z][a-z]+", body)[0]
        body = re.sub(r"\s+", " ", body).strip()
        if 20 <= len(body) <= 2000:
            items.append(body)
    return items


def clean_section(s: str) -> str:
    s = re.sub(r"NTSB/[A-Z]{3}-\d{2}/\d{2}", " ", s)  # running headers
    s = re.sub(r"\n\s*\d+\s*\n", "\n", s)  # bare page numbers
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def main() -> None:
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        r
        for r in csv.DictReader(open(MANIFEST))
        if r["extraction_status"] == "ok" and r["report_type"] in TYPES
    ]

    enriched = {}
    with open(ENRICHED) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("source") == "NTSB_REPORT":
                enriched[rec["record_id"]] = rec

    out, failures = [], []
    for r in rows:
        rn = r["report_number"]
        txt_path = REPO / r["txt_path"]
        if not txt_path.exists():
            failures.append({"report": rn, "reason": "txt missing"})
            continue
        text = txt_path.read_text(errors="ignore")

        parsed = split_aab(text) if r["report_type"] == "AAB" else split_aar(text)
        erec = enriched.get(rn)
        structured = (erec or {}).get("structured") or {}

        has_parsed_gold = parsed is not None and (
            parsed["findings_items"] or parsed["probable_cause_text"] or parsed.get("conclusions_text")
        )
        has_coded_gold = bool(structured.get("occurrences") or structured.get("findings"))
        if not has_parsed_gold and not has_coded_gold:
            failures.append({"report": rn, "reason": "no usable gold (parse and coded both empty)"})
            continue

        if parsed is not None:
            input_text = text[: parsed["split_at"]]
        else:
            input_text = text  # coded-only gold; nothing to withhold was found
        input_text = input_text[:MAX_INPUT_CHARS]
        if len(input_text) < MIN_INPUT_CHARS:
            failures.append({"report": rn, "reason": f"input too short ({len(input_text)} chars)"})
            continue
        (INPUTS_DIR / f"{rn}.txt").write_text(input_text)

        out.append(
            {
                "report_number": rn,
                "report_type": r["report_type"],
                "title": r["title"],
                "ntsb_accident_id": r["ntsb_accident_id"],
                "accident_date": r["accident_date"],
                "publication_date": r["publication_date"],
                "input_path": f"event_extraction/gold496/inputs/{rn}.txt",
                "input_chars": len(input_text),
                "gold": {
                    "findings_items": parsed["findings_items"] if parsed else [],
                    "probable_cause_text": parsed["probable_cause_text"] if parsed else "",
                    "conclusions_text": parsed.get("conclusions_text", "") if parsed else "",
                    "coded_occurrences": structured.get("occurrences") or [],
                    "coded_seq_of_events": structured.get("seq_of_events") or [],
                    "coded_findings": structured.get("findings") or [],
                    "coded_primary_cause": structured.get("primary_cause_text") or "",
                    "injury": structured.get("injury"),
                },
            }
        )

    with open(OUT_DIR / "gold.jsonl", "w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")

    n_pc = sum(1 for r in out if r["gold"]["probable_cause_text"])
    n_fnd = sum(1 for r in out if r["gold"]["findings_items"])
    n_occ = sum(1 for r in out if r["gold"]["coded_occurrences"])
    report = {
        "total_candidates": len(rows),
        "gold_records": len(out),
        "with_probable_cause": n_pc,
        "with_findings_items": n_fnd,
        "with_coded_occurrences": n_occ,
        "failures": failures,
        "by_type": {},
    }
    for t in sorted(TYPES):
        sub = [r for r in out if r["report_type"] == t]
        report["by_type"][t] = {
            "n": len(sub),
            "pc": sum(1 for r in sub if r["gold"]["probable_cause_text"]),
            "findings": sum(1 for r in sub if r["gold"]["findings_items"]),
        }
    with open(OUT_DIR / "build_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2)[:3000])


if __name__ == "__main__":
    main()
