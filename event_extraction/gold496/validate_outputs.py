"""Validate extraction outputs and sync state.json.

Walks outputs/<model>/<report>.json, runs the v4 schema validator on each,
and updates task statuses in state.json:
  pending -> done      (valid JSON, valid v4 chain)
  pending -> invalid   (file exists but fails validation; will be retried once)

Usage: python3 event_extraction/gold496/validate_outputs.py [--window N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "event_extraction" / "scripts"))
from validate_extraction import Report, load_def_enums, validate_v4  # noqa: E402

SCHEMA = REPO / "event_extraction" / "prompts" / "schema_v4.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=None, help="window number to stamp on newly validated tasks")
    args = ap.parse_args()

    enums = load_def_enums(SCHEMA)
    state_path = HERE / "state.json"
    state = json.loads(state_path.read_text())

    n_ok = n_bad = 0
    problems_log = []
    for model in state["models"]:
        outdir = HERE / "outputs" / model
        if not outdir.exists():
            continue
        for fp in sorted(outdir.glob("*.json")):
            rn = fp.stem
            key = f"{model}/{rn}"
            task = state["tasks"].get(key)
            if task is None or task["status"] == "done":
                continue
            try:
                raw = fp.read_text().strip()
                # tolerate accidental markdown fences
                if raw.startswith("```"):
                    raw = raw.strip("`")
                    raw = raw.split("\n", 1)[1] if raw.startswith("json") else raw
                rec = json.loads(raw)
                rep = Report()
                probs = validate_v4(rec, enums, rep)
                hard = [p for p in probs if "unknown" not in p.lower() or "factor_type" not in p]
            except Exception as e:  # noqa: BLE001
                rec, hard = None, [f"json parse error: {e}"]
            if rec is not None and not hard and isinstance(rec.get("chain"), list) and rec["chain"]:
                task["status"] = "done"
                if args.window is not None:
                    task["window"] = args.window
                n_ok += 1
            else:
                task["status"] = "invalid"
                task["attempts"] = task.get("attempts", 0) + 1
                n_bad += 1
                problems_log.append({"task": key, "problems": hard[:5]})

    state_path.write_text(json.dumps(state, indent=1))
    done = sum(1 for t in state["tasks"].values() if t["status"] == "done")
    pend = sum(1 for t in state["tasks"].values() if t["status"] == "pending")
    inv = sum(1 for t in state["tasks"].values() if t["status"] == "invalid")
    print(json.dumps({
        "validated_now": n_ok, "invalid_now": n_bad,
        "totals": {"done": done, "pending": pend, "invalid": inv},
        "problems": problems_log[:10],
    }, indent=1))


if __name__ == "__main__":
    main()
