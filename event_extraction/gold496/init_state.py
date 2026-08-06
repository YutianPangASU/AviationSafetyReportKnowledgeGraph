"""Initialize (or re-sync) state.json for the gold-496 extraction evaluation.

Creates one task per (report, model). Priority order: reports whose gold has
both a probable cause and findings-or-coded-sequence come first ("rich" gold),
shorter inputs before longer within each tier, so early windows complete the
most scoreable work.

Safe to re-run: existing task statuses are preserved; only new tasks are added.

Usage: python3 event_extraction/gold496/init_state.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = ["opus", "sonnet", "haiku"]
# starting per-window chunk sizes; the runner adapts these between windows
DEFAULT_CHUNK = {"opus": 12, "sonnet": 12, "haiku": 8}
RUN_DEADLINE = "2026-08-09T18:00:00-05:00"  # stop scheduling new work after this


def main() -> None:
    gold = [json.loads(l) for l in open(HERE / "gold.jsonl")]

    def rich(r):
        g = r["gold"]
        return bool(g["probable_cause_text"]) and bool(
            g["findings_items"] or g["coded_occurrences"]
        )

    ordered = sorted(gold, key=lambda r: (not rich(r), r["input_chars"]))
    order = [r["report_number"] for r in ordered]

    state_path = HERE / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        state = {
            "report_order": order,
            "models": MODELS,
            "chunk_size": dict(DEFAULT_CHUNK),
            "deadline": RUN_DEADLINE,
            "tasks": {},
            "runs": [],
        }
    state["report_order"] = order

    for model in MODELS:
        for rn in order:
            key = f"{model}/{rn}"
            state["tasks"].setdefault(
                key, {"status": "pending", "attempts": 0, "window": None}
            )

    state_path.write_text(json.dumps(state, indent=1))
    n = len(state["tasks"])
    pend = sum(1 for t in state["tasks"].values() if t["status"] == "pending")
    print(f"state.json ready: {n} tasks ({pend} pending), {len(order)} reports x {MODELS}")


if __name__ == "__main__":
    main()
