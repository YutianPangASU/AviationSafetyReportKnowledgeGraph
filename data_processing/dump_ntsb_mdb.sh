#!/usr/bin/env bash
# Dump key NTSB eADMS tables from Pre2008.mdb and avall.mdb to JSONL.
# These tables are stripped from data/corpus/corpus.jsonl and need to be
# rejoined for Stage 1 supervision and Stage 3 category buckets.
set -euo pipefail

ROOT="/home/yp6443/research/AviationSafetyReportKnowledgeGraph/data/NTSB_ASRS"
OUT="$ROOT/extracted"

# Tables to dump from each .mdb. Grouped by purpose:
#   core: per-accident structured records (joinable by ev_id [+ Aircraft_Key])
#   code: lookup tables for code -> description decoding
CORE=(events aircraft narratives Findings Occurrences seq_of_events injury Flight_Crew engines flight_time)
CODE=(ct_iaids ct_seqevt eADMSPUB_DataDictionary dt_events dt_aircraft dt_Flight_Crew Country states NTSB_Admin)

dump_db() {
    local db="$1" outdir="$2"
    mkdir -p "$outdir"
    for t in "${CORE[@]}" "${CODE[@]}"; do
        local f="$outdir/$t.jsonl"
        if [[ -s "$f" ]]; then
            echo "  skip $t (exists, $(wc -l < "$f") rows)"
            continue
        fi
        if mdb-json "$db" "$t" > "$f.tmp" 2>/dev/null; then
            mv "$f.tmp" "$f"
            echo "  $t -> $f.jsonl ($(wc -l < "$f") rows)"
        else
            rm -f "$f.tmp"
            echo "  $t MISSING in $db"
        fi
    done
}

echo "=== Pre2008.mdb ==="
dump_db "$ROOT/Pre2008.mdb" "$OUT/Pre2008"
echo
echo "=== avall.mdb ==="
dump_db "$ROOT/avall.mdb" "$OUT/avall"

echo
echo "Done. Sizes:"
du -sh "$OUT"/*/
