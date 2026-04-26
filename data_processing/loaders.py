"""Per-source loaders.

Each loader returns a ``pandas.DataFrame`` with a common schema:

    record_id, source, date, location, title, text, url, tail

``tail`` is an aircraft registration (e.g. ``N47BA``, ``C-GUDO``, ``F-GPAD``)
from the source's structured field when available; loaders that have no
structured tail column leave it blank and let the cross-source dedup stage
recover it via regex over the narrative.

``text`` is the raw narrative before cleaning or filtering. Callers should
then concatenate frames and run them through ``clean`` and ``dedupe``.
"""
from __future__ import annotations

import io
import os
import re
import subprocess
from pathlib import Path
from typing import Iterable, List

import pandas as pd

# --------------------------------------------------------------------------- #
# BEA
# --------------------------------------------------------------------------- #


def load_bea(csv_path: str | os.PathLike) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    out = pd.DataFrame(
        {
            "record_id": df["file_number"].astype(str),
            "source": "BEA",
            "date": df.get("date", ""),
            "location": df.get("location", "").str.cat(
                df.get("state_of_occurrence", ""), sep=", ", na_rep=""
            ),
            "title": df.get("title", ""),
            "text": df.get("summary", ""),
            "url": df.get("detail_url", ""),
            "tail": df.get("registration", "").astype(str),
        }
    )
    return out


# --------------------------------------------------------------------------- #
# NTSB ASRS (.mdb access databases)
# --------------------------------------------------------------------------- #


def _export_mdb_table(mdb_path: str, table: str) -> pd.DataFrame:
    csv_out = subprocess.check_output(["mdb-export", mdb_path, table])
    return pd.read_csv(
        io.StringIO(csv_out.decode("utf-8", errors="replace")),
        low_memory=False,
    )


def _list_mdb_tables(mdb_path: str) -> List[str]:
    out = subprocess.check_output(["mdb-tables", "-1", mdb_path]).decode(errors="replace")
    return [t for t in out.splitlines() if t.strip()]


def _load_ntsb_asrs_modern(mdb_path: str) -> pd.DataFrame:
    """Schema used by avall.mdb / Pre2008.mdb: a ``narratives`` table."""
    df = _export_mdb_table(mdb_path, "narratives")
    for col in ("narr_accp", "narr_accf", "narr_cause", "narr_inc"):
        if col not in df.columns:
            df[col] = ""
    df = df.fillna("")
    text = (
        df["narr_accp"].astype(str)
        + "  "
        + df["narr_accf"].astype(str)
        + "  "
        + df["narr_cause"].astype(str)
    ).str.strip()
    mask = text.str.len() == 0
    if mask.any():
        text = text.mask(mask, df["narr_inc"].astype(str))
    return pd.DataFrame(
        {
            "record_id": df["ev_id"].astype(str)
            + "_"
            + df.get("Aircraft_Key", pd.Series([""] * len(df))).astype(str),
            "source": f"NTSB_ASRS:{Path(mdb_path).stem}",
            "date": df["ev_id"].astype(str).str.slice(0, 8),
            "location": "",
            "title": "",
            "text": text,
            "url": "",
            "tail": "",
        }
    )


def _load_ntsb_asrs_pre1982(mdb_path: str) -> pd.DataFrame:
    """PRE1982.MDB schema: REMARKS + CAUSE live in tblSecondHalf, joined to
    tblFirstHalf on RecNum for event metadata."""
    second = _export_mdb_table(mdb_path, "tblSecondHalf")
    first_cols = ["RecNum", "DATE_OCCURRENCE", "LOCATION", "LOCAT_STATE_TERR", "DOCKET_NO"]
    try:
        first = _export_mdb_table(mdb_path, "tblFirstHalf")
        keep = [c for c in first_cols if c in first.columns]
        first = first[keep]
    except subprocess.CalledProcessError:
        first = pd.DataFrame(columns=first_cols)
    merged = second.merge(first, on="RecNum", how="left").fillna("")
    remarks = merged.get("REMARKS", pd.Series([""] * len(merged))).astype(str)
    cause = merged.get("CAUSE", pd.Series([""] * len(merged))).astype(str)
    text = (remarks + "  " + cause).str.strip()
    loc = (
        merged.get("LOCATION", pd.Series([""] * len(merged))).astype(str)
        + ", "
        + merged.get("LOCAT_STATE_TERR", pd.Series([""] * len(merged))).astype(str)
    ).str.strip(", ")
    return pd.DataFrame(
        {
            "record_id": merged.get("DOCKET_NO", merged["RecNum"]).astype(str),
            "source": f"NTSB_ASRS:{Path(mdb_path).stem}",
            "date": merged.get("DATE_OCCURRENCE", "").astype(str),
            "location": loc,
            "title": "",
            "text": text,
            "url": "",
            "tail": merged.get("REGIST_NO", pd.Series([""] * len(merged))).astype(str),
        }
    )


def load_ntsb_asrs(mdb_paths: Iterable[str | os.PathLike]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for p in mdb_paths:
        p = str(p)
        tables = _list_mdb_tables(p)
        if "narratives" in tables:
            frames.append(_load_ntsb_asrs_modern(p))
        elif "tblSecondHalf" in tables:
            frames.append(_load_ntsb_asrs_pre1982(p))
        else:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------- #
# NTSB_REPORTS (plain-text reports extracted from PDFs)
# --------------------------------------------------------------------------- #


def _read_text_file(p: Path) -> str:
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def load_ntsb_reports(
    manifest_csv: str | os.PathLike,
    txt_dir: str | os.PathLike,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_csv, dtype=str).fillna("")
    txt_dir = Path(txt_dir)
    texts: List[str] = []
    for rel in manifest["txt_path"]:
        candidate = Path(rel)
        if not candidate.is_absolute():
            candidate = Path(manifest_csv).parent.parent / rel
        if not candidate.exists():
            name = Path(rel).name
            candidate = txt_dir / name
        texts.append(_read_text_file(candidate) if candidate.exists() else "")
    out = pd.DataFrame(
        {
            "record_id": manifest["report_number"].astype(str),
            "source": "NTSB_REPORT",
            "date": manifest.get("accident_date", ""),
            "location": manifest.get("location", ""),
            "title": manifest.get("title", ""),
            "text": pd.Series(texts),
            "url": manifest.get("pdf_url", ""),
            "tail": "",
        }
    )
    return out


# --------------------------------------------------------------------------- #
# FAA AIDS (tab-delimited text, one file per year-range, 'a'=accident 'e'=incident)
# --------------------------------------------------------------------------- #


def _load_faa_aids_a_file(path: Path) -> pd.DataFrame:
    """Parse the structured a*.txt / e*.txt-sized record dump.

    The structured files expose ~180 coded columns; for corpus building we
    only need ``c5`` (record id), ``c9`` (date), ``c13``/``c14`` (state/city),
    plus ``c119`` (the truncated ~115-char remark). We use this file for
    metadata that will later be joined onto the richer e*-narrative.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        na_values=["\\N"],
        keep_default_na=False,
        on_bad_lines="skip",
        engine="python",
    ).fillna("")
    keep = {"c5": "record_id"}
    if "c5" not in df.columns:
        return pd.DataFrame()
    out = pd.DataFrame({"record_id": df["c5"].astype(str)})
    out["date"] = df.get("c9", pd.Series([""] * len(df))).astype(str)
    city = df.get("c14", pd.Series([""] * len(df)))
    state = df.get("c13", pd.Series([""] * len(df)))
    out["location"] = (city.astype(str) + ", " + state.astype(str)).str.strip(", ")
    out["c119"] = df.get("c119", pd.Series([""] * len(df))).astype(str)
    # c22 = aircraft "N" number / registration (AIDS codebook).
    out["tail"] = df.get("c22", pd.Series([""] * len(df))).astype(str)
    return out


def _load_faa_aids_e_file(path: Path) -> pd.DataFrame:
    """Parse the e*.txt narrative dump.

    Older years use ``c5\\tremark``; the 2020-26 export uses ``id\\trmks``.
    The narrative text is rich (often 200+ words) and is the primary source
    of FAA AIDS narratives.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        on_bad_lines="skip",
        engine="python",
    )
    cols = {c.lower(): c for c in df.columns}
    id_col = cols.get("c5") or cols.get("id")
    text_col = cols.get("remark") or cols.get("rmks")
    if id_col is None or text_col is None:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "record_id": df[id_col].astype(str),
            "text": df[text_col].astype(str),
        }
    )


def load_faa_aids(aids_dir: str | os.PathLike) -> pd.DataFrame:
    """Join structured a*.txt metadata onto the rich e*.txt narratives.

    Files that have no matching e*.txt narrative (if any) fall back to the
    truncated c119 remark so the record is still available for downstream
    length filtering.
    """
    aids_dir = Path(aids_dir)
    a_files = sorted(
        p for p in aids_dir.glob("a*.txt") if re.match(r"^a\d{4}_\d{2}\.txt$", p.name)
    )
    e_files = sorted(
        p for p in aids_dir.glob("e*.txt") if re.match(r"^e\d{4}_\d{2}\.txt$", p.name)
    )

    # Concatenate all e-file narratives keyed by record_id. The same id may
    # appear in multiple year-range dumps (e.g. 2020_25 + 2020_26 overlap);
    # we keep the longest narrative.
    e_frames = [_load_faa_aids_e_file(p) for p in e_files]
    e_frames = [f for f in e_frames if not f.empty]
    if e_frames:
        narr = pd.concat(e_frames, ignore_index=True)
        narr["wc"] = narr["text"].str.split().str.len().fillna(0)
        narr = (
            narr.sort_values("wc", ascending=False)
            .drop_duplicates(subset=["record_id"], keep="first")
            .drop(columns=["wc"])
        )
    else:
        narr = pd.DataFrame(columns=["record_id", "text"])

    a_frames = [_load_faa_aids_a_file(p) for p in a_files]
    a_frames = [f for f in a_frames if not f.empty]
    meta = (
        pd.concat(a_frames, ignore_index=True)
        if a_frames
        else pd.DataFrame(columns=["record_id", "date", "location", "c119"])
    )
    # Year bucket comes from the source a*.txt filename; record where each
    # came from so we can attribute stats per year range.
    bucket_map: List[str] = []
    for path, frame in zip(a_files, a_frames):
        bucket_map.extend([path.stem] * len(frame))
    if bucket_map:
        meta = meta.assign(bucket=bucket_map)

    merged = meta.merge(narr, on="record_id", how="left")
    merged["text"] = merged["text"].fillna("")
    fallback = merged["text"].str.len() == 0
    merged.loc[fallback, "text"] = merged.loc[fallback, "c119"]
    merged["source"] = "FAA_AIDS:" + merged.get("bucket", "").astype(str)
    out = pd.DataFrame(
        {
            "record_id": merged["record_id"].astype(str),
            "source": merged["source"],
            "date": merged.get("date", ""),
            "location": merged.get("location", ""),
            "title": "",
            "text": merged["text"].astype(str),
            "url": "",
            "tail": merged.get("tail", pd.Series([""] * len(merged))).astype(str),
        }
    )
    return out


# --------------------------------------------------------------------------- #
# TSB Canada (occurrence CSV with a Summary column)
# --------------------------------------------------------------------------- #


def load_tsb_canada(occurrence_csv: str | os.PathLike) -> pd.DataFrame:
    # The file starts with a UTF-8 BOM + some noise ("77u/\ufeff"). pandas
    # with encoding="utf-8-sig" strips the BOM; the stray prefix sits on the
    # first header cell which we rename by position.
    df = pd.read_csv(
        occurrence_csv,
        dtype=str,
        encoding="utf-8-sig",
        on_bad_lines="skip",
        low_memory=False,
    ).fillna("")
    first = df.columns[0]
    if first != "OccID":
        df = df.rename(columns={first: "OccID"})
    out = pd.DataFrame(
        {
            "record_id": df.get("OccNo", df["OccID"]).astype(str),
            "source": "TSB_CANADA",
            "date": df.get("OccDate", ""),
            "location": df.get("Location", "").astype(str),
            "title": df.get("CommonName", ""),
            "text": df.get("Summary", "").astype(str),
            "url": "",
            "tail": "",  # no structured column; extracted from narrative downstream
        }
    )
    return out
