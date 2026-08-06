# Gold-496 Extraction Evaluation — Runbook

Evaluates causal chain extraction accuracy of Claude models (opus, sonnet,
haiku) against 157 full NTSB investigation reports whose investigator
conclusions are withheld from the inputs. Gold answers (parsed Findings,
Probable Cause, coded NTSB database sequences) live in `gold.jsonl` and MUST
NEVER appear in any extraction prompt.

## How it runs (headless, survives session and terminal exits)

A detached driver executes one window every 6 hours from 2026-08-06 20:37
until 2026-08-09 18:00 CDT or until all 471 tasks are done. Each window:
select tasks per model from `state.json` (rich-gold, short-input reports
first; identical report sets across models), run each through
`claude -p --model <id> --tools ""` on the subscription, validate outputs
against the v4 schema, adapt per-model chunk sizes (grow ~25% on clean
windows, halve on quota errors, floor 4, cap 24), and log to `run.log`.

Launch (only if not already running; the pid file guards double starts):

    cd /home/yp6443/research/AviationSafetyReportKnowledgeGraph
    nohup python3 -u event_extraction/gold496/run_windows.py \
        >> event_extraction/gold496/run.log 2>&1 &

Monitor:

    tail -f event_extraction/gold496/run.log
    python3 -c "import json;from collections import Counter;\
s=json.load(open('event_extraction/gold496/state.json'));\
print(Counter(t['status'] for t in s['tasks'].values()));print(s['chunk_size'])"

Stop: `kill $(cat event_extraction/gold496/gold496.pid)`.

Recovery after a machine reboot or crash: just relaunch the nohup command;
state.json on disk resumes exactly where it left off.

## Files

- `build_gold.py` -> `gold.jsonl`, `inputs/` (regenerating is safe/idempotent)
- `init_state.py` -> `state.json` (preserves existing task statuses)
- `run_windows.py` — the detached driver (schedule, CLI calls, adaptation)
- `extraction_task.md` — v4 extraction specification (spec section is shared
  by the driver's CLI prompts; the Procedure section is for agent-based runs)
- `validate_outputs.py` — schema validation, marks tasks done/invalid
- `score_gold.py` — run manually AFTER extraction completes, with the local
  Qwen judge (cross-family): structural metrics, cause recall, node support
- `COMPLETED.md` — written by the driver at finish

## Hard rules

- Extraction prompts contain ONLY the spec and the input narrative — never
  gold.jsonl content, findings, probable cause, or coded data.
- Never modify `gold.jsonl`, `inputs/`, or `extraction_task.md` while runs
  are in flight.
- At most 3 attempts per task (driver and validator both increment).
- Scoring uses a judge from a different model family than the extractors.
