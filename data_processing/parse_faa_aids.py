"""Parse FAA AIDS A*.txt accident-main files into structured JSONL.

The A-files are TSV with a one-row header naming columns by code (c5, c1, ...).
Mapping comes from Afilelayout.txt; we extract only the columns relevant to
Stage 3 (per-category bucket) and Stage 4 (cause attribution) supervision.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path("/home/yp6443/research/AviationSafetyReportKnowledgeGraph/data/FAA_AIDS")
OUT = ROOT / "extracted"
OUT.mkdir(exist_ok=True)

# Columns to extract, code -> human field name. Codes from Afilelayout.txt.
KEEP = {
    "c5":   "record_id",                     # control number, joins to corpus.jsonl
    "c1":   "event_type",                    # A | I (accident vs incident)
    "c2":   "far_part",
    "c4":   "investigation_agency",
    "c9":   "event_date",
    "c10":  "event_time_local",
    "c75":  "midair_flag",
    "c95":  "phase_of_flight_text",
    "c96":  "phase_of_flight_code",
    "c77":  "cause_primary_text",            # primary cause factor text
    "c78":  "cause_primary_code",
    "c85":  "cause_secondary_text",
    "c86":  "cause_secondary_code",
    "c99":  "cause_general_category_text",
    "c100": "cause_general_category_code",
    "c161": "cause_additional_text",
    "c160": "cause_additional_code",
    "c163": "cause_2nd_additional_text",
    "c162": "cause_2nd_additional_code",
    "c105": "flying_condition_primary",      # VFR/IFR
    "c107": "flying_condition_secondary",
    "c109": "light_condition_text",
    "c117": "runway_condition_code",
    "c118": "braking_condition_code",
    "c233": "pilot_killed_flag",
    "c234": "second_pilot_killed_flag",
    "c23":  "aircraft_make",
    "c24":  "aircraft_model",
    "c34":  "engine_make",
    "c35":  "engine_model",
}

def parse_one(p: Path) -> tuple[int, int]:
    with p.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        # Build positional indexes for kept columns; stay forgiving if a column is missing
        idx = {col_code: header.index(col_code) for col_code in KEEP if col_code in header}
        out_path = OUT / f"{p.stem}.jsonl"
        n_in = n_out = 0
        with out_path.open("w") as outf:
            for line in f:
                n_in += 1
                row = line.rstrip("\n").split("\t")
                rec = {}
                for col_code, name in KEEP.items():
                    if col_code not in idx:
                        continue
                    i = idx[col_code]
                    if i < len(row):
                        v = row[i].strip()
                        rec[name] = v if v else None
                rec["source"] = f"FAA_AIDS:{p.stem}"  # mirrors corpus.jsonl source naming
                outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_out += 1
        return n_in, n_out

def main():
    a_files = sorted(ROOT.glob("a*.txt"))
    print(f"Found {len(a_files)} A-files")
    grand = 0
    for p in a_files:
        n_in, n_out = parse_one(p)
        print(f"  {p.name:20s} -> {OUT / (p.stem + '.jsonl')} ({n_out:,} rows)")
        grand += n_out
    print(f"TOTAL FAA AIDS records: {grand:,}")

if __name__ == "__main__":
    main()
