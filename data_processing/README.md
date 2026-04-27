# Data processing

## Stage 0 — corpus build (pre-existing)

| Script | Output |
|---|---|
| `loaders.py` | per-source raw loaders (BEA, FAA AIDS, NTSB ASRS/REPORTS, TSB) |
| `clean.py` | text normalization |
| `dedupe.py` | MinHash dedup, cross-source clustering |
| `build_corpus.py` | produces `data/corpus/corpus.jsonl` (178,013 records, narrative + minimal metadata) |

This produces a flat narrative corpus. **It does not preserve the structured
supervision fields** that downstream stages need — that is what the v3
redesign added below.

## Stage 0 — supervision rejoin (added 2026-04-27)

`data/corpus/corpus.jsonl` carries narrative text only. The structured fields
needed for Stage-1 supervision (`Findings`, `seq_of_events`, `Occurrences`)
and Stage-3 categorisation (CICTT-equivalent codes) were stripped during
corpus build. They live in the upstream `Pre2008.mdb`, `avall.mdb`, and FAA
AIDS `A*.txt` files. The rejoin pipeline:

```
data/NTSB_ASRS/{Pre2008,avall}.mdb     dump_ntsb_mdb.sh    -> data/NTSB_ASRS/extracted/{Pre2008,avall}/*.jsonl
data/FAA_AIDS/A*.txt                   parse_faa_aids.py   -> data/FAA_AIDS/extracted/*.jsonl
data/NTSB_REPORTS/manifest.csv         (loaded directly)
                       all of the above
                       +
data/corpus/corpus.jsonl    build_corpus_enriched.py    -> data/corpus/corpus_enriched.jsonl
```

| Script | What it does |
|---|---|
| [`dump_ntsb_mdb.sh`](dump_ntsb_mdb.sh) | Uses `mdb-tools` to dump every relevant table from `Pre2008.mdb` and `avall.mdb` to JSONL, one file per table per database. Idempotent — skips tables already dumped. |
| [`parse_faa_aids.py`](parse_faa_aids.py) | Parses FAA AIDS A-files (TSV with column-coded headers) into structured JSONL. Extracts only the columns relevant to Stage 3 (cause primary/secondary/general, phase of flight, flying conditions, aircraft fields). |
| [`build_corpus_enriched.py`](build_corpus_enriched.py) | Joins everything onto `corpus.jsonl` by `record_id` (which decomposes to NTSB `ev_id_<Aircraft_Key>` for ASRS sources or FAA `c5` control number for AIDS sources). Decodes integer codes via the `ct_seqevt` lookup table. Outputs `corpus_enriched.jsonl` with a `structured` payload per record. |

Run order:

```bash
bash data_processing/dump_ntsb_mdb.sh            # ~5 min
python data_processing/parse_faa_aids.py         # ~1 min
python data_processing/build_corpus_enriched.py  # ~2 min
```

Coverage: 145k of 178k records (82%) carry structured supervision after
the join. BEA (1.7k) and TSB Canada (31k) are narrative-only — the upstream
sources do not expose structured fields.

See [docs/2026-04-27-v3-schema-redesign.md](../docs/2026-04-27-v3-schema-redesign.md) §"Root 1" for design rationale.

## Schema discoveries (worth knowing)

- **Cause attribution lives in different tables across NTSB databases.** Pre2008
  (older taxonomy) puts cause flags in `seq_of_events.Cause_Factor='C'`;
  avall (post-2008 CAST) puts them in the dense `Findings` table with
  HFACS-style `category/subcategory/section/subsection/modifier` codes.
- **`record_id` decomposition.** For NTSB ASRS sources, `record_id` is
  `<ev_id>_<Aircraft_Key>` (e.g. `20001213X25852_1`). For FAA AIDS,
  `record_id` is the `c5` control number (e.g. `19940722040519I`).
- **Source name `NTSB_ASRS` is misleading.** `Pre2008.mdb` and `avall.mdb` are
  actually the NTSB eADMS structured **accident** database, not ASRS
  (NASA's anonymous **incident** reporting system). The corpus naming
  convention came from the original loader script and is now load-bearing
  in record_ids.
