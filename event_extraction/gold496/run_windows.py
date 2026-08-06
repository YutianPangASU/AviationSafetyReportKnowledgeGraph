"""Detached driver for the gold-496 extraction evaluation.

Runs independently of any Claude Code session (start with nohup). Every six
hours it executes one evaluation window: select pending (model, report)
tasks from state.json, run each through the headless Claude CLI
(`claude -p --model <id> --tools ""`, subscription-billed), validate the
outputs, adapt per-model chunk sizes, and log. Stops when all tasks are done
or the deadline passes.

Launch:
  cd /home/yp6443/research/AviationSafetyReportKnowledgeGraph
  nohup python3 -u event_extraction/gold496/run_windows.py \
      >> event_extraction/gold496/run.log 2>&1 &

Monitor:
  tail -f event_extraction/gold496/run.log
  python3 -c "import json;s=json.load(open('event_extraction/gold496/state.json'));\
from collections import Counter;print(Counter(t['status'] for t in s['tasks'].values()))"

Flags:
  --dry-run   select and print this window's tasks, call nothing, change nothing
  --now       run the first window immediately instead of waiting for the schedule
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

ANCHOR = datetime(2026, 8, 6, 20, 37)          # first window, local time
PERIOD = timedelta(hours=6)
DEADLINE = datetime(2026, 8, 9, 18, 0)
MODEL_IDS = {"opus": "claude-opus-5", "sonnet": "claude-sonnet-5", "haiku": "claude-haiku-4-5"}
WORKERS = 3
CALL_TIMEOUT = 1800
MAX_ATTEMPTS = 3
QUOTA_MARKERS = ("usage limit", "rate limit", "limit reached", "out of usage", "quota", "overloaded")

SPEC_MARKER = "## Extraction Specification (v4)"


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def load_state() -> dict:
    return json.loads((HERE / "state.json").read_text())


def save_state(state: dict) -> None:
    (HERE / "state.json").write_text(json.dumps(state, indent=1))


def build_prompt(rn: str) -> str:
    task_md = (HERE / "extraction_task.md").read_text()
    spec = task_md[task_md.index(SPEC_MARKER):]
    narrative = (HERE / "inputs" / f"{rn}.txt").read_text(errors="ignore")
    return (
        "You are performing causal chain extraction from ONE full NTSB "
        "investigation report. Follow the specification below exactly.\n\n"
        f"{spec}\n\n"
        f"# Input narrative (report {rn})\n\n{narrative}\n\n"
        "# Output\n\nReply with ONLY the single JSON object "
        '{"chain": [...], "outcome_severity": "..."}. '
        "No markdown fences, no commentary, no tool use."
    )


def select_tasks(state: dict) -> dict[str, list[str]]:
    picked: dict[str, list[str]] = {}
    for model in state["models"]:
        chunk = state["chunk_size"].get(model, 8)
        chosen: list[str] = []
        for status_wanted in ("invalid", "pending"):
            for rn in state["report_order"]:
                if len(chosen) >= chunk:
                    break
                t = state["tasks"].get(f"{model}/{rn}")
                if (
                    t
                    and t["status"] == status_wanted
                    and t.get("attempts", 0) < MAX_ATTEMPTS
                    and rn not in chosen
                ):
                    chosen.append(rn)
        picked[model] = chosen
    return picked


def run_one(model: str, rn: str) -> dict:
    out_path = HERE / "outputs" / model / f"{rn}.json"
    try:
        prompt = build_prompt(rn)
        proc = subprocess.run(
            ["claude", "-p", "--model", MODEL_IDS[model], "--tools", ""],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=CALL_TIMEOUT,
            cwd=str(REPO),
        )
        text = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        blob = (text + " " + err).lower()
        if proc.returncode != 0 or not text:
            quota = any(m in blob for m in QUOTA_MARKERS)
            return {"model": model, "rn": rn, "ok": False, "quota": quota,
                    "err": (err or text)[:300]}
        out_path.write_text(text)
        return {"model": model, "rn": rn, "ok": True, "quota": False, "err": ""}
    except subprocess.TimeoutExpired:
        return {"model": model, "rn": rn, "ok": False, "quota": False, "err": "timeout"}
    except Exception as e:  # noqa: BLE001
        return {"model": model, "rn": rn, "ok": False, "quota": False, "err": str(e)[:300]}


def run_window(window_n: int, dry: bool = False) -> None:
    state = load_state()
    picked = select_tasks(state)
    total = sum(len(v) for v in picked.values())
    log(f"window {window_n}: selected " +
        ", ".join(f"{m}:{len(v)}" for m, v in picked.items()) + f" ({total} tasks)")
    if dry:
        for m, v in picked.items():
            log(f"  DRY {m}: {v}")
        return
    if total == 0:
        return

    jobs = [(m, rn) for m, rns in picked.items() for rn in rns]
    results = []
    quota_hit: set[str] = set()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(run_one, m, rn): (m, rn) for m, rn in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            tag = "ok" if r["ok"] else ("QUOTA" if r["quota"] else "fail")
            log(f"  {r['model']}/{r['rn']}: {tag}" + (f" ({r['err']})" if r["err"] else ""))
            if r["quota"]:
                quota_hit.add(r["model"])

    # validate everything written this window (updates task statuses)
    subprocess.run(
        [sys.executable, str(HERE / "validate_outputs.py"), "--window", str(window_n)],
        cwd=str(REPO),
    )

    # driver-side bookkeeping: attempts for CLI failures, chunk adaptation
    state = load_state()
    for r in results:
        if not r["ok"]:
            t = state["tasks"].get(f"{r['model']}/{r['rn']}")
            if t is not None:
                t["attempts"] = t.get("attempts", 0) + 1
    per_model = {}
    for model in state["models"]:
        mres = [r for r in results if r["model"] == model]
        n_ok = sum(1 for r in mres if r["ok"])
        per_model[model] = {"attempted": len(mres), "cli_ok": n_ok}
        cur = state["chunk_size"].get(model, 8)
        if model in quota_hit:
            state["chunk_size"][model] = max(4, cur // 2)
        elif mres and n_ok == len(mres):
            state["chunk_size"][model] = min(24, cur + max(1, cur // 4))
    state["runs"].append({
        "window": window_n,
        "started": datetime.now().isoformat(timespec="seconds"),
        "per_model": per_model,
        "quota_hit": sorted(quota_hit),
    })
    save_state(state)
    done = sum(1 for t in state["tasks"].values() if t["status"] == "done")
    log(f"window {window_n} complete: {done}/{len(state['tasks'])} tasks done; "
        f"chunk sizes now {state['chunk_size']}")


def all_finished(state: dict) -> bool:
    return not any(
        t["status"] in ("pending", "invalid") and t.get("attempts", 0) < MAX_ATTEMPTS
        for t in state["tasks"].values()
    )


def finalize() -> None:
    subprocess.run(
        [sys.executable, str(HERE / "validate_outputs.py")], cwd=str(REPO)
    )
    state = load_state()
    from collections import Counter

    counts = Counter(t["status"] for t in state["tasks"].values())
    per_model = {
        m: Counter(
            t["status"] for k, t in state["tasks"].items() if k.startswith(m + "/")
        )
        for m in state["models"]
    }
    lines = [
        "# Gold-496 Extraction Evaluation - Completed",
        "",
        f"Finished: {datetime.now().isoformat(timespec='seconds')}",
        f"Windows run: {len(state['runs'])}",
        f"Totals: {dict(counts)}",
        "",
    ] + [f"- {m}: {dict(c)}" for m, c in per_model.items()] + [
        "",
        "Next step: score with `python3 event_extraction/gold496/score_gold.py "
        "--judge-endpoint <local qwen endpoint>` (cross-family judge).",
    ]
    (HERE / "COMPLETED.md").write_text("\n".join(lines) + "\n")
    log("finalized; COMPLETED.md written")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--now", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        run_window(window_n=len(load_state()["runs"]) + 1, dry=True)
        return

    pidfile = HERE / "gold496.pid"
    if pidfile.exists():
        old = pidfile.read_text().strip()
        if old and Path(f"/proc/{old}").exists():
            log(f"another driver is running (pid {old}); exiting")
            return
    pidfile.write_text(str(os.getpid()))

    log(f"driver started (pid {os.getpid()}); anchor {ANCHOR}, deadline {DEADLINE}")
    try:
        while True:
            now = datetime.now()
            if now >= DEADLINE or all_finished(load_state()):
                finalize()
                break
            if args.now:
                nxt = now
                args.now = False
            else:
                k = 0
                nxt = ANCHOR
                while nxt <= now:
                    k += 1
                    nxt = ANCHOR + k * PERIOD
            wait = (nxt - datetime.now()).total_seconds()
            if wait > 0:
                log(f"sleeping {wait/3600:.2f} h until window at {nxt}")
                time.sleep(wait)
            if datetime.now() >= DEADLINE:
                finalize()
                break
            run_window(window_n=len(load_state()["runs"]) + 1)
    finally:
        pidfile.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
