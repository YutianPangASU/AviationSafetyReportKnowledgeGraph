"""Build a unified, cleaned, de-duplicated aviation safety corpus.

Usage (from the repo root):

    python -m data_processing.build_corpus \
        --data-root data \
        --out-dir data/corpus \
        --min-words 30

Outputs:

    data/corpus/corpus.jsonl        one record per line (kept after filtering)
    data/corpus/dropped.jsonl       records dropped, with reason field
    data/corpus/stats.json          per-source counts and filter breakdown
    data/corpus/clusters.jsonl      near-duplicate cluster memberships
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .clean import has_causal_link, normalize_text, word_count
from .dedupe import deduplicate_minhash
from .loaders import (
    load_bea,
    load_faa_aids,
    load_ntsb_asrs,
    load_ntsb_reports,
    load_tsb_canada,
)


COMMON_COLS = ["record_id", "source", "date", "location", "title", "text", "url"]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_all(data_root: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    bea_csv = data_root / "BEA" / "bea_notified_events.csv"
    if bea_csv.exists():
        _log(f"loading BEA from {bea_csv.name}")
        frames.append(load_bea(bea_csv))

    asrs_dir = data_root / "NTSB_ASRS"
    mdb_files = sorted(asrs_dir.glob("*.[Mm][Dd][Bb]"))
    if mdb_files:
        _log(f"loading NTSB_ASRS: {[p.name for p in mdb_files]}")
        frames.append(load_ntsb_asrs(mdb_files))

    ntsb_manifest = data_root / "NTSB_REPORTS" / "manifest.csv"
    ntsb_txt = data_root / "NTSB_REPORTS" / "txt"
    if ntsb_manifest.exists() and ntsb_txt.exists():
        _log(f"loading NTSB_REPORTS from {ntsb_manifest}")
        frames.append(load_ntsb_reports(ntsb_manifest, ntsb_txt))

    aids_dir = data_root / "FAA_AIDS"
    if aids_dir.exists():
        _log(f"loading FAA_AIDS from {aids_dir}")
        frames.append(load_faa_aids(aids_dir))

    tsb_csv = data_root / "TSB_CANADA" / "ASISdb_MDOTW_VW_OCCURRENCE_PUBLIC.csv"
    if tsb_csv.exists():
        _log(f"loading TSB_CANADA from {tsb_csv.name}")
        frames.append(load_tsb_canada(tsb_csv))

    if not frames:
        raise RuntimeError(f"no data sources found under {data_root}")
    merged = pd.concat(frames, ignore_index=True)
    for col in COMMON_COLS:
        if col not in merged.columns:
            merged[col] = ""
    return merged[COMMON_COLS].astype(str)


def clean_and_filter(
    df: pd.DataFrame, min_words: int, require_causal: bool
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, int]]]:
    _log(f"normalizing text for {len(df):,} rows")
    df = df.copy()
    df["text"] = df["text"].map(normalize_text)
    df["wc"] = df["text"].map(word_count)

    stats: Dict[str, Dict[str, int]] = {}
    for src, grp in df.groupby("source"):
        stats[src] = {"loaded": int(len(grp))}

    empty_mask = df["wc"] == 0
    short_mask = (~empty_mask) & (df["wc"] < min_words)
    if require_causal:
        long_enough = df["wc"] >= min_words
        causal_mask = long_enough & ~df["text"].map(has_causal_link)
    else:
        causal_mask = pd.Series(False, index=df.index)

    drop_reason = pd.Series("", index=df.index)
    drop_reason[empty_mask] = "empty"
    drop_reason[short_mask] = f"short(<{min_words}w)"
    drop_reason[causal_mask] = "no_causal_marker"

    dropped = df[drop_reason != ""].copy()
    dropped["drop_reason"] = drop_reason[drop_reason != ""]
    kept = df[drop_reason == ""].copy()

    for src in stats:
        sub = df[df["source"] == src]
        stats[src]["empty"] = int((sub.index.isin(df.index[empty_mask])).sum())
        stats[src]["short"] = int((sub.index.isin(df.index[short_mask])).sum())
        stats[src]["no_causal_marker"] = int(
            (sub.index.isin(df.index[causal_mask])).sum()
        )
        stats[src]["kept_pre_dedup"] = int(
            (sub.index.isin(kept.index)).sum()
        )
    return kept, dropped, stats


def dedupe(
    df: pd.DataFrame,
    threshold: float,
    num_perm: int,
    shingle_k: int,
) -> tuple[pd.DataFrame, List[List[int]]]:
    _log(
        f"minhash-lsh dedup: n={len(df):,} threshold={threshold} "
        f"perm={num_perm} k={shingle_k}"
    )
    clusters = deduplicate_minhash(
        df["text"].tolist(),
        threshold=threshold,
        num_perm=num_perm,
        shingle_k=shingle_k,
    )
    # Pick a representative per cluster: longest narrative, break ties by
    # source priority (full NTSB reports > BEA summaries > TSB/ASRS > FAA AIDS).
    priority = {
        "NTSB_REPORT": 0,
        "BEA": 1,
        "TSB_CANADA": 2,
    }

    def src_rank(src: str) -> int:
        if src.startswith("NTSB_ASRS"):
            return 3
        if src.startswith("FAA_AIDS"):
            return 4
        return priority.get(src, 5)

    keep_idx: List[int] = []
    wcs = df["text"].map(word_count).tolist()
    srcs = df["source"].tolist()
    for cluster in clusters:
        if len(cluster) == 1:
            keep_idx.append(cluster[0])
            continue
        best = max(
            cluster,
            key=lambda i: (wcs[i], -src_rank(srcs[i])),
        )
        keep_idx.append(best)
    deduped = df.iloc[sorted(keep_idx)].reset_index(drop=True)
    return deduped, clusters


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="data", type=Path)
    ap.add_argument("--out-dir", default="data/corpus", type=Path)
    ap.add_argument("--min-words", default=30, type=int)
    ap.add_argument(
        "--no-causal-filter",
        action="store_true",
        help="keep narratives that meet the length threshold even if no "
        "reason / effect marker is present.",
    )
    ap.add_argument("--dedup-threshold", default=0.8, type=float)
    ap.add_argument("--num-perm", default=128, type=int)
    ap.add_argument("--shingle-k", default=5, type=int)
    args = ap.parse_args(argv)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = load_all(args.data_root)
    _log(f"merged rows: {len(merged):,} across {merged['source'].nunique()} sources")

    kept, dropped, stats = clean_and_filter(
        merged, min_words=args.min_words, require_causal=not args.no_causal_filter
    )
    _log(f"after length+causal filter: {len(kept):,} kept, {len(dropped):,} dropped")

    deduped, clusters = dedupe(
        kept,
        threshold=args.dedup_threshold,
        num_perm=args.num_perm,
        shingle_k=args.shingle_k,
    )

    for src, grp in kept.groupby("source"):
        kept_ids = set(deduped.index.tolist())
        # `deduped` was reset_index, so compare by (source, record_id) instead.
    kept_pairs = set(zip(deduped["source"], deduped["record_id"]))
    for src in stats:
        sub = kept[kept["source"] == src]
        kept_after = sum(
            1 for sr, rid in zip(sub["source"], sub["record_id"]) if (sr, rid) in kept_pairs
        )
        stats[src]["kept_final"] = int(kept_after)
        stats[src]["duplicates_removed"] = int(
            stats[src]["kept_pre_dedup"] - kept_after
        )

    _log(f"after dedup: {len(deduped):,} unique narratives")

    corpus_path = out_dir / "corpus.jsonl"
    dropped_path = out_dir / "dropped.jsonl"
    stats_path = out_dir / "stats.json"
    clusters_path = out_dir / "clusters.jsonl"

    deduped_out = deduped.drop(columns=["wc"], errors="ignore")
    deduped_out.to_json(corpus_path, orient="records", lines=True, force_ascii=False)
    dropped.drop(columns=["wc"], errors="ignore").to_json(
        dropped_path, orient="records", lines=True, force_ascii=False
    )
    with open(clusters_path, "w", encoding="utf-8") as f:
        src = kept["source"].tolist()
        rid = kept["record_id"].tolist()
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            f.write(
                json.dumps(
                    {
                        "size": len(cluster),
                        "members": [{"source": src[i], "record_id": rid[i]} for i in cluster],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    with open(stats_path, "w", encoding="utf-8") as f:
        summary = {
            "totals": {
                "merged": int(len(merged)),
                "kept_pre_dedup": int(len(kept)),
                "kept_final": int(len(deduped)),
                "dropped": int(len(dropped)),
                "dup_clusters": sum(1 for c in clusters if len(c) > 1),
            },
            "by_source": stats,
            "params": {
                "min_words": args.min_words,
                "causal_filter": not args.no_causal_filter,
                "dedup_threshold": args.dedup_threshold,
                "num_perm": args.num_perm,
                "shingle_k": args.shingle_k,
            },
        }
        json.dump(summary, f, indent=2)
    _log(f"wrote {corpus_path}, {dropped_path}, {clusters_path}, {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
