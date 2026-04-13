# NTSB Published Reports Scraper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `data/scrape_ntsb_reports.py`, a four-phase scraper that enumerates every NTSB published aviation report, downloads each PDF, extracts text via `pdfplumber`, and emits a manifest CSV joinable to `avall.mdb` by NTSB accident ID.

**Architecture:** Single Python script with four sequentially-runnable phases — Playwright enumeration → `requests` PDF download → `pdfplumber` text extraction → manifest assembly. Each phase is independently resumable from disk state. Matches the structure of `data/scrape_bea.py`.

**Tech Stack:** Python 3.10 (existing `ntsb` conda env), `playwright` (new), `pdfplumber` (new), `requests` (existing), standard library.

**Spec:** [docs/superpowers/specs/2026-04-11-ntsb-reports-scraper-design.md](../specs/2026-04-11-ntsb-reports-scraper-design.md)

---

## File Structure

| File | Purpose |
|---|---|
| `data/scrape_ntsb_reports.py` | Main scraper — all four phases as functions, plus `main()` and CLI parser |
| `data/test_ntsb_reports_extract.py` | Pytest-style unit tests for the accident-ID regex (no network) |
| `data/NTSB_REPORTS/` | Output directory (gitignored) — `pdf/`, `txt/`, `listing.json`, `manifest.csv`, error logs, `.playwright/` |
| `.gitignore` | Add `/data/NTSB_REPORTS` |
| `README.md` | New "NTSB Published Reports" subsection under "Data Accessibility" |

The single-file script convention matches `scrape_bea.py` and `download_faa_aids.py`. No package decomposition needed.

---

## Task 1: Project setup — directories, gitignore, dependency install, script skeleton

**Files:**
- Create: `data/scrape_ntsb_reports.py`
- Modify: `.gitignore`
- Bash: install playwright + pdfplumber, install Chromium

- [ ] **Step 1: Add the new data source to `.gitignore`**

Open `.gitignore` and add `/data/NTSB_REPORTS` after the existing `/data/FAA_AIDS` line so the block becomes:

```
/data/NTSB_ASRS
/data/TSB_CANADA
/data/FAA_AIDS
/data/NTSB_REPORTS
```

- [ ] **Step 2: Install new Python dependencies into the `ntsb` conda environment**

Run:
```bash
conda activate ntsb && pip install playwright pdfplumber pytest
```

Then install Chromium for Playwright (one-time, ~300 MB):
```bash
PLAYWRIGHT_BROWSERS_PATH=$PWD/data/NTSB_REPORTS/.playwright python -m playwright install chromium
```

Expected: `playwright` and `pdfplumber` install without error; Chromium downloads to `data/NTSB_REPORTS/.playwright/`.

- [ ] **Step 3: Create the empty output directory tree**

Run:
```bash
mkdir -p data/NTSB_REPORTS/pdf data/NTSB_REPORTS/txt
```

- [ ] **Step 4: Create `data/scrape_ntsb_reports.py` with module docstring, imports, and constants**

```python
"""
Scrape all published NTSB aviation accident/safety reports from
https://www.ntsb.gov/investigations/AccidentReports/Pages/Reports.aspx

Four-phase pipeline:
  Phase 1  Enumerate          Playwright loads Reports.aspx, filters Aviation,
                              paginates, harvests metadata + PDF URLs.
  Phase 2  Download           Fetches each PDF via requests, with retry/resume.
  Phase 3  Extract            pdfplumber -> .txt files.
  Phase 4  Manifest           Builds manifest.csv keyed by NTSB accident ID.

Output: data/NTSB_REPORTS/

Usage:
  python data/scrape_ntsb_reports.py                    # full run / resume
  python data/scrape_ntsb_reports.py --limit 5          # smoke test
  python data/scrape_ntsb_reports.py --delay 2          # custom delay
  python data/scrape_ntsb_reports.py --phase 1          # single phase
  python data/scrape_ntsb_reports.py --force-extract    # re-extract all txt
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import requests

# ----- Paths -----
OUTPUT_DIR = "data/NTSB_REPORTS"
PDF_DIR = os.path.join(OUTPUT_DIR, "pdf")
TXT_DIR = os.path.join(OUTPUT_DIR, "txt")
LISTING_FILE = os.path.join(OUTPUT_DIR, "listing.json")
MANIFEST_FILE = os.path.join(OUTPUT_DIR, "manifest.csv")
FAILED_DOWNLOADS_LOG = os.path.join(OUTPUT_DIR, "failed_downloads.log")
FAILED_EXTRACTIONS_LOG = os.path.join(OUTPUT_DIR, "failed_extractions.log")
PHASE1_DOM_DUMP = os.path.join(OUTPUT_DIR, "phase1_dom_dump.html")

REPORTS_URL = "https://www.ntsb.gov/investigations/AccidentReports/Pages/Reports.aspx"

# ----- NTSB accident ID format: 3 letters + 2 digits + 2 letters + 3 digits -----
# Examples: DCA19MA086, ERA22FA123, WPR21LA047
ACCIDENT_ID_RE = re.compile(r"\b[A-Z]{3}\d{2}[A-Z]{2}\d{3}\b")

# ----- Manifest schema (15 columns, see spec §3) -----
MANIFEST_FIELDS = [
    "report_number",
    "report_type",
    "title",
    "publication_date",
    "accident_date",
    "location",
    "ntsb_accident_id",
    "accident_id_source",
    "pdf_url",
    "pdf_path",
    "txt_path",
    "pdf_pages",
    "txt_chars",
    "extraction_status",
    "scraped_at",
]

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def main():
    pass  # filled in Task 7


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Verify the script imports cleanly**

Run:
```bash
conda activate ntsb && python -c "import data.scrape_ntsb_reports" 2>&1 || python data/scrape_ntsb_reports.py
```

Expected: no errors, no output (`main()` is empty).

- [ ] **Step 6: Commit**

```bash
git add .gitignore data/scrape_ntsb_reports.py
git commit -m "scaffold NTSB reports scraper: skeleton, gitignore, deps"
```

---

## Task 2: TDD the accident-ID regex extraction function

The regex itself is the only code path with enough logic to unit-test. Write the test first.

**Files:**
- Create: `data/test_ntsb_reports_extract.py`
- Modify: `data/scrape_ntsb_reports.py`

- [ ] **Step 1: Write the failing test file**

Create `data/test_ntsb_reports_extract.py`:

```python
"""Unit tests for the NTSB accident-ID regex extraction.

Run with: pytest data/test_ntsb_reports_extract.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from scrape_ntsb_reports import extract_accident_id_from_text


# Real NTSB cover-page snippets (paraphrased; format preserved).
SAMPLES = [
    # (description, input_text, expected_accident_id)
    (
        "AAR-23-01 cover",
        "Aircraft Accident Report\nNTSB/AAR-23/01\nAccident Number: DCA19MA086\nLas Vegas, Nevada",
        "DCA19MA086",
    ),
    (
        "AAB-21-04 cover",
        "Aircraft Accident Brief\nNTSB Accident ID: ERA20FA123\nDate: April 5, 2020",
        "ERA20FA123",
    ),
    (
        "ID embedded in paragraph",
        "On May 15, 2021, about 0930 PDT, accident WPR21LA047 occurred near...",
        "WPR21LA047",
    ),
    (
        "ID in upper-case header",
        "NTSB IDENTIFICATION: CEN18FA200",
        "CEN18FA200",
    ),
    (
        "Hyphenated near-misses ignored",
        "Accident: ANC-22-LA-001 (this is not the standard format)",
        None,
    ),
    (
        "Lowercase ignored (must be uppercase)",
        "accident dca19ma086 lowercase",
        None,
    ),
    (
        "First match wins among multiple",
        "Initial: ERA20FA123. Related: WPR21LA047.",
        "ERA20FA123",
    ),
    (
        "Surrounded by punctuation",
        "(DCA19MA086).",
        "DCA19MA086",
    ),
    (
        "Empty string",
        "",
        None,
    ),
    (
        "No ID anywhere",
        "Aircraft Accident Report on the loss of N12345 on January 1, 2020.",
        None,
    ),
]


def test_accident_id_extraction():
    failures = []
    for description, text, expected in SAMPLES:
        actual = extract_accident_id_from_text(text)
        if actual != expected:
            failures.append(f"  {description!r}: expected {expected!r}, got {actual!r}")
    assert not failures, "Accident ID extraction failures:\n" + "\n".join(failures)
```

- [ ] **Step 2: Run the test to verify it fails (function not yet defined)**

Run:
```bash
conda activate ntsb && pytest data/test_ntsb_reports_extract.py -v
```

Expected: FAIL with `ImportError: cannot import name 'extract_accident_id_from_text'`.

- [ ] **Step 3: Implement `extract_accident_id_from_text` in `scrape_ntsb_reports.py`**

Add this function below the constants block (above `main()`):

```python
def extract_accident_id_from_text(text):
    """Return the first NTSB accident ID found in `text`, or None.

    NTSB accident IDs match the pattern: 3 uppercase letters + 2 digits +
    2 uppercase letters + 3 digits, e.g. DCA19MA086, ERA20FA123.
    """
    if not text:
        return None
    match = ACCIDENT_ID_RE.search(text)
    return match.group(0) if match else None
```

- [ ] **Step 4: Run the test again to verify it passes**

Run:
```bash
conda activate ntsb && pytest data/test_ntsb_reports_extract.py -v
```

Expected: PASS — `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add data/test_ntsb_reports_extract.py data/scrape_ntsb_reports.py
git commit -m "add accident-ID regex extraction with unit tests"
```

---

## Task 3: Phase 1 — Playwright enumeration of Reports.aspx aviation listings

This phase has inherent DOM-discovery uncertainty: the exact CSS selectors for the aviation filter button, listing rows, and pagination controls cannot be known until Playwright actually loads the page. The implementation includes a debug-mode escape hatch and a DOM dump on selector failure (per spec §5).

**Files:**
- Modify: `data/scrape_ntsb_reports.py`

- [ ] **Step 1: Add the Phase 1 function with DOM-exploration scaffold**

Add this function after `extract_accident_id_from_text`:

```python
def enumerate_listing(force=False):
    """Phase 1: Use Playwright to load Reports.aspx, filter to Aviation,
    paginate, and harvest every report's metadata + PDF URL.

    Caches result to LISTING_FILE. Skips work if cache exists and force=False.
    Returns the list of listing entries.
    """
    if os.path.exists(LISTING_FILE) and not force:
        with open(LISTING_FILE, "r", encoding="utf-8") as f:
            entries = json.load(f)
        print(f"[Phase 1] Loaded {len(entries)} entries from cache: {LISTING_FILE}", flush=True)
        return entries

    # Lazy import so the script can run Phase 4 alone without Playwright installed.
    from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout

    print("[Phase 1] Launching Playwright Chromium...", flush=True)
    os.environ.setdefault(
        "PLAYWRIGHT_BROWSERS_PATH",
        os.path.abspath(os.path.join(OUTPUT_DIR, ".playwright")),
    )

    entries = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent=USER_AGENT)
        page = context.new_page()

        try:
            print(f"[Phase 1] Loading {REPORTS_URL}", flush=True)
            page.goto(REPORTS_URL, wait_until="networkidle", timeout=60000)

            # Wait for the dynamic listing to render. The SharePoint app shows
            # a "loading" spinner first; we wait for actual report rows.
            try:
                page.wait_for_selector("a[href$='.pdf'], .ms-listviewtable tr", timeout=30000)
            except PwTimeout:
                _dump_dom_and_fail(page, "initial listing did not render")

            # Apply Aviation filter. NTSB exposes mode tabs/links; the exact
            # selector is discovered on first run -- if it changes, the DOM
            # dump in failed_dom_dump.html shows the current structure.
            try:
                aviation = page.get_by_role("link", name=re.compile("Aviation", re.I)).first
                if aviation.count() > 0:
                    aviation.click()
                    page.wait_for_load_state("networkidle", timeout=30000)
            except Exception as e:
                print(f"[Phase 1] Aviation filter click failed: {e}", flush=True)
                _dump_dom_and_fail(page, "aviation filter not clickable")

            # Paginate and harvest
            page_num = 1
            while True:
                print(f"[Phase 1] Harvesting page {page_num}...", flush=True)
                page_entries = _harvest_current_page(page)
                if not page_entries and page_num == 1:
                    _dump_dom_and_fail(page, "no entries found on page 1")
                entries.extend(page_entries)
                print(f"[Phase 1]   collected {len(page_entries)} entries (total {len(entries)})", flush=True)

                next_btn = page.get_by_role("link", name=re.compile(r"^next$", re.I)).first
                if next_btn.count() == 0 or not next_btn.is_visible():
                    break
                try:
                    next_btn.click()
                    page.wait_for_load_state("networkidle", timeout=30000)
                except Exception:
                    break
                page_num += 1
                time.sleep(1.0)

        finally:
            browser.close()

    # Deduplicate by pdf_url (in case a row is repeated across pages)
    seen = set()
    deduped = []
    for e in entries:
        if e["pdf_url"] in seen:
            continue
        seen.add(e["pdf_url"])
        deduped.append(e)

    print(f"[Phase 1] Done. {len(deduped)} unique entries (after dedup).", flush=True)
    with open(LISTING_FILE, "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2)
    print(f"[Phase 1] Saved to {LISTING_FILE}", flush=True)
    return deduped


def _harvest_current_page(page):
    """Extract listing entries from the currently-rendered page.

    Each entry: {report_number, report_type, title, publication_date,
                 accident_date, location, pdf_url}.
    """
    entries = []
    # Strategy: find every PDF anchor on the page, then pull surrounding row
    # metadata. The SharePoint listing renders rows with a PDF link plus
    # adjacent cells for title/date/location.
    pdf_links = page.locator("a[href$='.pdf']").all()
    for link in pdf_links:
        try:
            href = link.get_attribute("href") or ""
            if not href:
                continue
            if href.startswith("/"):
                href = "https://www.ntsb.gov" + href
            elif not href.startswith("http"):
                continue

            title = (link.text_content() or "").strip()

            # Walk up to the row container and pull all cell text
            row = link.locator("xpath=ancestor::tr[1]")
            row_text = ""
            if row.count() > 0:
                row_text = (row.first.text_content() or "").strip()

            report_number = _infer_report_number_from_url(href)
            report_type = _infer_report_type(report_number, title)
            pub_date = _extract_date(row_text)
            location = _extract_location(row_text, title)

            entries.append({
                "report_number": report_number,
                "report_type": report_type,
                "title": title,
                "publication_date": pub_date,
                "accident_date": "",  # filled in Phase 4 from PDF text if possible
                "location": location,
                "pdf_url": href,
            })
        except Exception as e:
            print(f"[Phase 1]   row error: {e}", flush=True)
            continue
    return entries


def _infer_report_number_from_url(url):
    """e.g. .../Reports/AAR2301.pdf -> AAR-23-01"""
    base = os.path.splitext(os.path.basename(url))[0]
    m = re.match(r"([A-Z]+)(\d{2})(\d{2,3})$", base)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return base


def _infer_report_type(report_number, title):
    rn_upper = (report_number or "").upper()
    title_upper = (title or "").upper()
    for code in ("AAR", "AAB", "SIR", "SS", "SRR", "SR"):
        if rn_upper.startswith(code) or code in title_upper:
            return code if code != "SR" else "SRR"
    return "OTHER"


def _extract_date(text):
    """Find the first ISO or US date in `text`. Returns YYYY-MM-DD or ''."""
    if not text:
        return ""
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
    if m:
        return m.group(0)
    m = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", text)
    if m:
        mm, dd, yy = m.groups()
        return f"{yy}-{int(mm):02d}-{int(dd):02d}"
    months = "January|February|March|April|May|June|July|August|September|October|November|December"
    m = re.search(rf"\b({months})\s+(\d{{1,2}}),?\s+(\d{{4}})\b", text)
    if m:
        month_name, dd, yy = m.groups()
        month_num = "January February March April May June July August September October November December".split().index(month_name) + 1
        return f"{yy}-{month_num:02d}-{int(dd):02d}"
    return ""


def _extract_location(row_text, title):
    """Best-effort: pull a comma-separated location like 'Dallas, Texas' from
    the row text or title. Returns the longest match or ''."""
    if not row_text and not title:
        return ""
    candidates = re.findall(r"[A-Z][a-zA-Z\.]+(?:\s+[A-Z][a-zA-Z\.]+)*,\s+[A-Z][a-zA-Z]+", (row_text or "") + " " + (title or ""))
    if not candidates:
        return ""
    return max(candidates, key=len)


def _dump_dom_and_fail(page, reason):
    """Save the current page HTML to PHASE1_DOM_DUMP and raise."""
    try:
        with open(PHASE1_DOM_DUMP, "w", encoding="utf-8") as f:
            f.write(page.content())
        print(f"[Phase 1] DOM dumped to {PHASE1_DOM_DUMP} for inspection.", flush=True)
    except Exception:
        pass
    raise RuntimeError(f"Phase 1 failed: {reason}")
```

- [ ] **Step 2: Smoke-run Phase 1 alone**

Run:
```bash
conda activate ntsb && rm -f data/NTSB_REPORTS/listing.json && python -c "
import sys; sys.path.insert(0, 'data')
from scrape_ntsb_reports import enumerate_listing
entries = enumerate_listing()
print(f'OK: {len(entries)} entries')
print('First 3:', entries[:3])
"
```

Expected outcomes (handle either):
- **Success path:** `data/NTSB_REPORTS/listing.json` exists, contains > 50 entries, the first few have non-empty `pdf_url` and `title`. Move to Step 3.
- **DOM-mismatch failure:** `data/NTSB_REPORTS/phase1_dom_dump.html` is created and the script raises. Open the dump file, identify the actual selectors used by NTSB's current page (search for `<a href` patterns containing `.pdf`, look for the pagination control, look for the aviation filter element), and adjust the selectors in `_harvest_current_page` and the aviation-filter / next-button locators in `enumerate_listing` accordingly. Re-run.

If the DOM-dump path is hit twice with no progress, switch the Playwright launch to `headless=False` temporarily (visual debugging) and rerun once to see the page interactively.

- [ ] **Step 3: Spot-check the listing**

Run:
```bash
python -c "
import json
with open('data/NTSB_REPORTS/listing.json') as f:
    entries = json.load(f)
print(f'Total entries: {len(entries)}')
print('Sample:')
for e in entries[:5]:
    print(f'  {e[\"report_number\"]:20s} {e[\"report_type\"]:6s} {e[\"pdf_url\"]}')
print('PDF URLs unique:', len(set(e['pdf_url'] for e in entries)) == len(entries))
print('Reports with non-empty title:', sum(1 for e in entries if e['title']))
"
```

Expected: total > 50, all pdf_urls end in `.pdf`, all unique, > 80% have non-empty titles.

- [ ] **Step 4: Commit Phase 1**

```bash
git add data/scrape_ntsb_reports.py
git commit -m "implement Phase 1: Playwright enumeration of NTSB reports listing"
```

---

## Task 4: Phase 2 — PDF download with retry, resume, and consecutive-error guard

**Files:**
- Modify: `data/scrape_ntsb_reports.py`

- [ ] **Step 1: Add the Phase 2 function**

Add this function below the Phase 1 helpers:

```python
def download_pdfs(entries, delay=1.5, limit=None):
    """Phase 2: Download each entry's PDF to PDF_DIR. Skips files already on
    disk. Retries 3x on HTTP error. Aborts after 5 consecutive failures.

    Returns the count of newly downloaded PDFs.
    """
    if limit is not None:
        # Sort by publication_date desc, take top `limit`
        sorted_entries = sorted(
            entries,
            key=lambda e: e.get("publication_date", "") or "",
            reverse=True,
        )
        entries = sorted_entries[:limit]
        print(f"[Phase 2] --limit {limit}: restricted to {len(entries)} most recent entries", flush=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    failed = []
    consecutive_errors = 0
    new_count = 0

    for i, entry in enumerate(entries):
        pdf_path = os.path.join(PDF_DIR, f"{entry['report_number']}.pdf")
        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
            continue

        url = entry["pdf_url"]
        success = False
        for attempt in range(3):
            try:
                resp = session.get(url, timeout=60)
                resp.raise_for_status()
                with open(pdf_path, "wb") as f:
                    f.write(resp.content)
                size_kb = len(resp.content) / 1024
                print(f"[Phase 2] [{i+1}/{len(entries)}] {entry['report_number']:20s} {size_kb:6.0f} KB", flush=True)
                success = True
                consecutive_errors = 0
                new_count += 1
                break
            except Exception as e:
                wait = 1 * (4 ** attempt)
                print(f"[Phase 2]   attempt {attempt+1} for {entry['report_number']} failed: {e}; sleeping {wait}s", flush=True)
                time.sleep(wait)

        if not success:
            failed.append(url)
            consecutive_errors += 1
            with open(FAILED_DOWNLOADS_LOG, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now(timezone.utc).isoformat()}\t{entry['report_number']}\t{url}\n")
            if consecutive_errors >= 5:
                print(f"[Phase 2] 5 consecutive failures, aborting. Re-run to resume.", flush=True)
                break

        time.sleep(delay)

    print(f"[Phase 2] Done. {new_count} new PDFs downloaded. {len(failed)} failed.", flush=True)
    if failed:
        print(f"[Phase 2] Failed URLs logged to {FAILED_DOWNLOADS_LOG}", flush=True)
    return new_count
```

- [ ] **Step 2: Smoke-run Phase 2 with limit=3**

Run:
```bash
conda activate ntsb && python -c "
import sys; sys.path.insert(0, 'data')
from scrape_ntsb_reports import enumerate_listing, download_pdfs
entries = enumerate_listing()
download_pdfs(entries, limit=3)
import os
pdfs = sorted(os.listdir('data/NTSB_REPORTS/pdf'))
print('PDFs on disk:', pdfs)
for p in pdfs:
    sz = os.path.getsize('data/NTSB_REPORTS/pdf/' + p)
    print(f'  {p}: {sz//1024} KB')
"
```

Expected: 3 PDFs land in `data/NTSB_REPORTS/pdf/`, each > 50 KB.

- [ ] **Step 3: Verify resume works**

Re-run the same command. Expected: `[Phase 2] Done. 0 new PDFs downloaded.` (because all 3 already exist on disk).

- [ ] **Step 4: Commit Phase 2**

```bash
git add data/scrape_ntsb_reports.py
git commit -m "implement Phase 2: PDF download with retry and resume"
```

---

## Task 5: Phase 3 — pdfplumber text extraction

**Files:**
- Modify: `data/scrape_ntsb_reports.py`

- [ ] **Step 1: Add the Phase 3 function**

Add this function below `download_pdfs`:

```python
def extract_text(force=False):
    """Phase 3: Run pdfplumber over each PDF in PDF_DIR, write joined text
    to TXT_DIR. Skip if txt is newer than pdf and force=False.

    Returns dict {report_number: extraction_status}.
    """
    import pdfplumber

    statuses = {}
    pdfs = sorted(f for f in os.listdir(PDF_DIR) if f.endswith(".pdf"))
    print(f"[Phase 3] Extracting text from {len(pdfs)} PDFs...", flush=True)

    for i, pdf_name in enumerate(pdfs):
        report_number = pdf_name[:-4]
        pdf_path = os.path.join(PDF_DIR, pdf_name)
        txt_path = os.path.join(TXT_DIR, f"{report_number}.txt")

        if os.path.exists(txt_path) and not force:
            if os.path.getmtime(txt_path) > os.path.getmtime(pdf_path):
                # Up to date; classify by content size
                size = os.path.getsize(txt_path)
                statuses[report_number] = "ok" if size >= 500 else "ocr_needed"
                continue

        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages_text = []
                for page in pdf.pages:
                    t = page.extract_text() or ""
                    pages_text.append(t)
                full_text = "\n\n".join(pages_text)

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(full_text)

            if len(full_text) < 500:
                statuses[report_number] = "ocr_needed"
                with open(FAILED_EXTRACTIONS_LOG, "a", encoding="utf-8") as f:
                    f.write(f"{datetime.now(timezone.utc).isoformat()}\t{report_number}\tneeds_ocr\t{len(full_text)} chars\n")
                print(f"[Phase 3] [{i+1}/{len(pdfs)}] {report_number}: only {len(full_text)} chars (ocr_needed)", flush=True)
            else:
                statuses[report_number] = "ok"
                print(f"[Phase 3] [{i+1}/{len(pdfs)}] {report_number}: {len(full_text)} chars", flush=True)

        except Exception as e:
            print(f"[Phase 3] [{i+1}/{len(pdfs)}] {report_number}: extraction failed: {e}", flush=True)
            statuses[report_number] = "failed"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("")
            with open(FAILED_EXTRACTIONS_LOG, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now(timezone.utc).isoformat()}\t{report_number}\tfailed\t{e}\n")

    print(f"[Phase 3] Done.", flush=True)
    return statuses
```

- [ ] **Step 2: Smoke-run Phase 3**

Run:
```bash
conda activate ntsb && python -c "
import sys; sys.path.insert(0, 'data')
from scrape_ntsb_reports import extract_text
statuses = extract_text()
import os
txts = sorted(os.listdir('data/NTSB_REPORTS/txt'))
print('TXTs:', txts)
for t in txts:
    sz = os.path.getsize('data/NTSB_REPORTS/txt/' + t)
    print(f'  {t}: {sz} chars, status={statuses.get(t[:-4])}')
"
```

Expected: 3 txt files exist (one per PDF from Task 4), all status `ok` with > 1000 chars.

- [ ] **Step 3: Eyeball one extracted text**

Run:
```bash
ls data/NTSB_REPORTS/txt/ | head -1 | xargs -I{} head -50 data/NTSB_REPORTS/txt/{}
```

Inspect: does the text look readable, with paragraphs and an accident ID visible somewhere on the cover? If you see garbled mojibake or text in the wrong order (column interleaving), note it for later improvement; do not block on it.

- [ ] **Step 4: Commit Phase 3**

```bash
git add data/scrape_ntsb_reports.py
git commit -m "implement Phase 3: pdfplumber text extraction with status tracking"
```

---

## Task 6: Phase 4 — Manifest builder with three-tier accident-ID extraction

**Files:**
- Modify: `data/scrape_ntsb_reports.py`

- [ ] **Step 1: Add Phase 4 helpers — accident-ID extraction from PDF first pages and avall.mdb fallback**

Add these helpers below `extract_text`:

```python
def extract_accident_id_from_pdf_first_pages(pdf_path, n_pages=2):
    """Open the PDF and run the regex over the first n_pages of text only.
    Returns the first match or None."""
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            chunks = []
            for page in pdf.pages[:n_pages]:
                chunks.append(page.extract_text() or "")
            return extract_accident_id_from_text("\n".join(chunks))
    except Exception:
        return None


def extract_accident_id_from_listing(entry):
    """Look for an accident ID in the listing entry's title and report_number."""
    for field in ("title", "report_number"):
        v = entry.get(field) or ""
        match = extract_accident_id_from_text(v)
        if match:
            return match
    return None


def extract_accident_id_from_mdb(entry):
    """Last-resort: query avall.mdb for an event matching this report's
    accident_date and location. Returns ev_id or None. Gracefully no-ops if
    mdbtools or avall.mdb is unavailable.
    """
    mdb_path = "data/NTSB_ASRS/avall.mdb"
    if not os.path.exists(mdb_path):
        return None
    accident_date = entry.get("accident_date") or entry.get("publication_date") or ""
    location = entry.get("location") or ""
    if not accident_date or not location:
        return None
    try:
        import subprocess
        # Export the events table to CSV via mdb-export, parse manually
        result = subprocess.run(
            ["mdb-export", mdb_path, "events"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            return None
        import io
        reader = csv.DictReader(io.StringIO(result.stdout))
        candidates = []
        loc_lower = location.lower()
        for row in reader:
            ev_date = (row.get("ev_date") or "")[:10]
            if ev_date != accident_date:
                continue
            ev_city = (row.get("ev_city") or "").lower()
            ev_state = (row.get("ev_state") or "").lower()
            if ev_city and ev_city in loc_lower or ev_state and ev_state in loc_lower:
                candidates.append(row.get("ev_id"))
        if len(candidates) == 1:
            return candidates[0]
        return None
    except Exception:
        return None
```

- [ ] **Step 2: Add the `build_manifest` function**

Add this function below the helpers:

```python
def build_manifest():
    """Phase 4: Walk pdf/ + txt/ + listing.json, extract accident IDs from
    PDFs / listing / mdb fallback, and rewrite manifest.csv.
    """
    if not os.path.exists(LISTING_FILE):
        print(f"[Phase 4] No listing.json yet. Run Phase 1 first.", flush=True)
        return
    with open(LISTING_FILE, "r", encoding="utf-8") as f:
        entries = json.load(f)
    by_report = {e["report_number"]: e for e in entries}

    rows = []
    pdfs = sorted(f for f in os.listdir(PDF_DIR) if f.endswith(".pdf"))
    print(f"[Phase 4] Building manifest for {len(pdfs)} PDFs...", flush=True)

    for i, pdf_name in enumerate(pdfs):
        report_number = pdf_name[:-4]
        pdf_path = os.path.join(PDF_DIR, pdf_name)
        txt_path = os.path.join(TXT_DIR, f"{report_number}.txt")

        entry = by_report.get(report_number, {
            "report_number": report_number,
            "report_type": "OTHER",
            "title": "",
            "publication_date": "",
            "accident_date": "",
            "location": "",
            "pdf_url": "",
        })

        # Three-tier accident-ID extraction
        acc_id = extract_accident_id_from_pdf_first_pages(pdf_path)
        acc_src = "pdf_first_page" if acc_id else None
        if not acc_id:
            acc_id = extract_accident_id_from_listing(entry)
            if acc_id:
                acc_src = "listing"
        if not acc_id:
            acc_id = extract_accident_id_from_mdb(entry)
            if acc_id:
                acc_src = "inferred"
        if not acc_id:
            acc_src = "missing"

        # PDF page count
        pdf_pages = 0
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                pdf_pages = len(pdf.pages)
        except Exception:
            pass

        # TXT char count + extraction status
        txt_chars = 0
        extraction_status = "failed"
        if os.path.exists(txt_path):
            txt_chars = os.path.getsize(txt_path)
            if txt_chars >= 500:
                extraction_status = "ok"
            elif txt_chars > 0:
                extraction_status = "ocr_needed"

        rows.append({
            "report_number": report_number,
            "report_type": entry.get("report_type", "OTHER"),
            "title": entry.get("title", ""),
            "publication_date": entry.get("publication_date", ""),
            "accident_date": entry.get("accident_date", ""),
            "location": entry.get("location", ""),
            "ntsb_accident_id": acc_id or "",
            "accident_id_source": acc_src,
            "pdf_url": entry.get("pdf_url", ""),
            "pdf_path": os.path.relpath(pdf_path),
            "txt_path": os.path.relpath(txt_path),
            "pdf_pages": pdf_pages,
            "txt_chars": txt_chars,
            "extraction_status": extraction_status,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
        })

        if (i + 1) % 25 == 0 or (i + 1) == len(pdfs):
            print(f"[Phase 4]   {i+1}/{len(pdfs)} processed", flush=True)

    with open(MANIFEST_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    n_with_id = sum(1 for r in rows if r["ntsb_accident_id"])
    print(f"[Phase 4] Done. {len(rows)} rows, {n_with_id} with accident ID. Saved to {MANIFEST_FILE}", flush=True)
```

- [ ] **Step 3: Smoke-run Phase 4**

Run:
```bash
conda activate ntsb && python -c "
import sys; sys.path.insert(0, 'data')
from scrape_ntsb_reports import build_manifest
build_manifest()
"
```

Then inspect:
```bash
python -c "
import csv
with open('data/NTSB_REPORTS/manifest.csv') as f:
    rows = list(csv.DictReader(f))
print(f'Manifest rows: {len(rows)}')
print('Columns:', list(rows[0].keys()) if rows else 'EMPTY')
for r in rows:
    print(f'  {r[\"report_number\"]:15s} type={r[\"report_type\"]:6s} acc_id={r[\"ntsb_accident_id\"]:12s} src={r[\"accident_id_source\"]:14s} status={r[\"extraction_status\"]}')
"
```

Expected: 3 rows (from earlier smoke test), each with all 15 columns. At least 2 of 3 should have a non-blank `ntsb_accident_id` with `accident_id_source=pdf_first_page` (1 miss tolerated).

- [ ] **Step 4: Commit Phase 4**

```bash
git add data/scrape_ntsb_reports.py
git commit -m "implement Phase 4: manifest builder with three-tier accident-ID extraction"
```

---

## Task 7: Wire phases together in `main()` with full CLI

**Files:**
- Modify: `data/scrape_ntsb_reports.py`

- [ ] **Step 1: Replace the empty `main()` with the full CLI driver**

Replace `def main(): pass` with:

```python
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None,
                        help="Phase 2 only: download just the N most recent reports (smoke test)")
    parser.add_argument("--delay", type=float, default=1.5,
                        help="Delay between PDF downloads in seconds (default: 1.5)")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4], default=None,
                        help="Run only the specified phase (1=enumerate, 2=download, 3=extract, 4=manifest)")
    parser.add_argument("--force-extract", action="store_true",
                        help="Phase 3: re-extract all txt files even if up to date")
    parser.add_argument("--force-enumerate", action="store_true",
                        help="Phase 1: re-enumerate even if listing.json exists")
    args = parser.parse_args()

    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(TXT_DIR, exist_ok=True)

    phases_to_run = [args.phase] if args.phase else [1, 2, 3, 4]

    entries = None
    if 1 in phases_to_run:
        entries = enumerate_listing(force=args.force_enumerate)
    if 2 in phases_to_run:
        if entries is None:
            if not os.path.exists(LISTING_FILE):
                print("[main] Cannot run Phase 2: listing.json missing. Run Phase 1 first.", flush=True)
                sys.exit(1)
            with open(LISTING_FILE, "r", encoding="utf-8") as f:
                entries = json.load(f)
        download_pdfs(entries, delay=args.delay, limit=args.limit)
    if 3 in phases_to_run:
        extract_text(force=args.force_extract)
    if 4 in phases_to_run:
        build_manifest()

    print("[main] All requested phases complete.", flush=True)
```

- [ ] **Step 2: Run the unit test set to make sure nothing regressed**

Run:
```bash
conda activate ntsb && pytest data/test_ntsb_reports_extract.py -v
```

Expected: PASS.

- [ ] **Step 3: End-to-end smoke test via the CLI**

Wipe the smoke output and re-run from a clean state, restricted to 5 reports:

```bash
rm -rf data/NTSB_REPORTS/pdf data/NTSB_REPORTS/txt data/NTSB_REPORTS/manifest.csv data/NTSB_REPORTS/listing.json data/NTSB_REPORTS/failed_*.log
mkdir -p data/NTSB_REPORTS/pdf data/NTSB_REPORTS/txt
conda activate ntsb && python data/scrape_ntsb_reports.py --limit 5
```

Verify:
```bash
ls data/NTSB_REPORTS/pdf/ | wc -l
ls data/NTSB_REPORTS/txt/ | wc -l
wc -l data/NTSB_REPORTS/manifest.csv
python -c "
import csv
with open('data/NTSB_REPORTS/manifest.csv') as f:
    rows = list(csv.DictReader(f))
n = len(rows)
n_with_id = sum(1 for r in rows if r['ntsb_accident_id'])
n_ok = sum(1 for r in rows if r['extraction_status'] == 'ok')
print(f'rows={n}, with_acc_id={n_with_id}, extract_ok={n_ok}')
assert n == 5, f'expected 5 rows, got {n}'
assert n_ok >= 4, f'expected >=4 ok extractions, got {n_ok}'
assert n_with_id >= 4, f'expected >=4 acc_ids, got {n_with_id}'
print('SMOKE TEST PASSED')
"
```

Expected: 5 PDFs, 5 txt files, 6 lines in manifest (header + 5), and the assertion line prints `SMOKE TEST PASSED`.

- [ ] **Step 4: Commit**

```bash
git add data/scrape_ntsb_reports.py
git commit -m "wire NTSB scraper phases into main() with full CLI"
```

---

## Task 8: README update — document the new data source

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the data tree in README**

In `README.md`, find the `data/` tree block (around line 18) and add a new entry for `NTSB_REPORTS/`:

Old:
```
data/
├── NTSB_ASRS/       # US NTSB Aviation Safety Reporting System
│   └── avall.mdb    # MS Access database with narratives, events, aircraft, etc.
├── TSB_CANADA/       # Transport Safety Board of Canada
```

New:
```
data/
├── NTSB_ASRS/       # US NTSB Aviation Safety Reporting System
│   └── avall.mdb    # MS Access database with narratives, events, aircraft, etc.
├── NTSB_REPORTS/    # US NTSB published narrative investigation reports
│   ├── pdf/         # Source PDFs (AAR, AAB, SIR, SS, SRR)
│   ├── txt/         # pdfplumber-extracted text (primary downstream input)
│   ├── listing.json # Phase 1 enumeration cache
│   └── manifest.csv # Per-report manifest, joinable to avall.mdb by ntsb_accident_id
├── TSB_CANADA/       # Transport Safety Board of Canada
```

- [ ] **Step 2: Add a "NTSB Published Reports" subsection**

Insert this section after the existing "NTSB data (US)" subsection (after the `mdbtools` paragraph):

```markdown
### NTSB Published Reports (US, narratives)
Source: https://www.ntsb.gov/investigations/AccidentReports/Pages/Reports.aspx

The full long-form NTSB published reports — Aircraft Accident Reports (AAR), Aircraft Accident Briefs (AAB), Special Investigation Reports (SIR), Safety Studies (SS), and Safety Recommendation Reports (SRR). These are the highest-richness narratives in the NTSB universe and are joinable back to `avall.mdb` rows by `ntsb_accident_id`.

Requires Playwright (one-time install ~300 MB):
```
conda activate ntsb
pip install playwright pdfplumber
PLAYWRIGHT_BROWSERS_PATH=$PWD/data/NTSB_REPORTS/.playwright python -m playwright install chromium
```

To re-scrape:
```
conda activate ntsb
python data/scrape_ntsb_reports.py            # full run / resume
python data/scrape_ntsb_reports.py --limit 5  # smoke test
```

Output: `data/NTSB_REPORTS/pdf/*.pdf`, `data/NTSB_REPORTS/txt/*.txt`, `data/NTSB_REPORTS/manifest.csv`. Manifest columns: report_number, report_type, title, publication_date, accident_date, location, ntsb_accident_id, accident_id_source, pdf_url, pdf_path, txt_path, pdf_pages, txt_chars, extraction_status, scraped_at.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "document NTSB published reports data source in README"
```

---

## Task 9: Full production run

This task runs the scraper without `--limit` to fetch every published aviation report. It is the final deliverable. Expect this to take a while (hundreds of PDFs at 1.5 s/each + extraction + manifest build).

**Files:** none (writes to `data/NTSB_REPORTS/`)

- [ ] **Step 1: Clean smoke-test artifacts and run for real**

```bash
rm -rf data/NTSB_REPORTS/pdf data/NTSB_REPORTS/txt data/NTSB_REPORTS/manifest.csv data/NTSB_REPORTS/listing.json data/NTSB_REPORTS/failed_*.log
mkdir -p data/NTSB_REPORTS/pdf data/NTSB_REPORTS/txt
conda activate ntsb && python data/scrape_ntsb_reports.py 2>&1 | tee data/NTSB_REPORTS/scrape_run.log
```

- [ ] **Step 2: Validate the full run**

```bash
python -c "
import csv, os
pdfs = [f for f in os.listdir('data/NTSB_REPORTS/pdf') if f.endswith('.pdf')]
txts = [f for f in os.listdir('data/NTSB_REPORTS/txt') if f.endswith('.txt')]
with open('data/NTSB_REPORTS/manifest.csv') as f:
    rows = list(csv.DictReader(f))
print(f'PDFs: {len(pdfs)}')
print(f'TXTs: {len(txts)}')
print(f'Manifest rows: {len(rows)}')
print(f'  with accident_id: {sum(1 for r in rows if r[\"ntsb_accident_id\"])}')
print(f'  extraction_status==ok: {sum(1 for r in rows if r[\"extraction_status\"]==\"ok\")}')
print(f'  ocr_needed: {sum(1 for r in rows if r[\"extraction_status\"]==\"ocr_needed\")}')
print(f'  failed: {sum(1 for r in rows if r[\"extraction_status\"]==\"failed\")}')
print('Report types:')
from collections import Counter
for t, c in Counter(r['report_type'] for r in rows).most_common():
    print(f'  {t}: {c}')
"
```

Expected: PDFs count == TXTs count == manifest row count. Most rows have `extraction_status=ok`. AAR/AAB/SIR rows should have high accident-ID coverage (>90%); SS/SRR rows are expected to have blank IDs.

- [ ] **Step 3: Manual quality spot check**

Pick 3 random AAR txt files and eyeball them vs. their PDFs. Check: cover page text present, narrative readable, no obvious column-interleaving garbage. If a systematic issue is found, file it as a follow-up — do not block the manifest deliverable.

- [ ] **Step 4: Final commit**

The PDFs and txts are gitignored, so this commit is just the run log if you want to preserve it (otherwise skip):

```bash
git add data/NTSB_REPORTS/scrape_run.log 2>/dev/null || true
git status  # confirm what would be committed
# If anything's worth committing:
# git commit -m "first full production run of NTSB reports scraper"
```

---

## Spec coverage check

| Spec section | Implemented in task |
|---|---|
| §1 Motivation, scope decisions | Reflected in tasks 3–6 (aviation only, all report types tagged, PDF+TXT+manifest, accident-ID linkage, Playwright) |
| §2.1 Four phases | Tasks 3, 4, 5, 6 |
| §2.2 Output directory layout | Task 1 (mkdir) + Task 8 (README) |
| §2.3 Dependencies | Task 1 (install) + Task 8 (README) |
| §3 Manifest schema (15 columns) | Task 1 (constants) + Task 6 (build_manifest) |
| §4 Three-tier accident-ID extraction | Task 2 (regex + tests) + Task 6 (three-tier helper functions) |
| §5 Error handling | Task 3 (DOM dump), Task 4 (retry + 5-error guard + log), Task 5 (extraction status + log) |
| §6 Testing strategy | Task 2 (regex unit test), Task 7 step 3 (smoke test), Task 9 step 3 (manual validation) |
| §7 CLI surface | Task 7 (full CLI in main()) |
| §8 Integration with ACE-Graph | Out of scope for this script; manifest schema ensures join compatibility |
| §9 Risks | Mitigations baked into Task 3 (DOM dump), Task 5 (ocr_needed status), Task 4 (resume) |
| §10 Deliverables | Task 1 (scaffold, gitignore), Task 2 (test file), Task 8 (README), Task 9 (first run) |

All spec requirements have a corresponding task. No placeholders. Method/function names used in later tasks (`extract_accident_id_from_text`, `enumerate_listing`, `download_pdfs`, `extract_text`, `build_manifest`, `extract_accident_id_from_pdf_first_pages`, `extract_accident_id_from_listing`, `extract_accident_id_from_mdb`) all match their definitions.
