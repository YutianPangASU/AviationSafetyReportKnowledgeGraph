# NTSB Published Reports Scraper — Design

**Status:** Approved in brainstorming 2026-04-11. Ready for implementation planning.
**Working title:** `scrape_ntsb_reports.py`
**One-line summary:** A four-phase scraper that enumerates every published NTSB aviation accident/safety report from `ntsb.gov/investigations/AccidentReports/Pages/Reports.aspx`, downloads each PDF, extracts text via `pdfplumber`, and emits a manifest CSV keyed by NTSB accident ID so the resulting narratives plug directly into the unified UASC corpus described in `2026-04-10-ace-graph-design.md`.

## 1. Motivation

### 1.1 Why this source

The existing NTSB data in this project (`data/NTSB_ASRS/avall.mdb`) is the structured Microsoft Access dump from `app.ntsb.gov/avdata`. It is rich in coded fields (events, aircraft, findings) but the narratives it carries (`narr_accp`, `narr_accf`, `narr_cause`) are *summary* fields, not the full investigative report.

The Reports.aspx page lists NTSB's **published narrative investigation reports** — the long-form documents that NTSB writes for the most consequential accidents. These include:

| Type | Description | Typical length |
|---|---|---|
| **AAR** Aircraft Accident Report | Full board-adopted investigation | 100–300 pp |
| **AAB** Aircraft Accident Brief | Shorter narrative for less complex accidents | 10–50 pp |
| **SIR** Special Investigation Report | Deep dive on a specific incident | 50–150 pp |
| **SS** Safety Study | Cross-cutting analysis (e.g. helicopter EMS operations) | 50–200 pp |
| **SRR** Safety Recommendation Report | Policy-style document | 20–80 pp |

Where they exist, these are the highest-richness narratives in the NTSB universe — the ground truth that ACE-Graph's Stage 1 hybrid event extraction was designed to operate on. Their absence from the current corpus is a gap.

### 1.2 Why now

The ACE-Graph design (`docs/superpowers/specs/2026-04-10-ace-graph-design.md`, Stage 0) defines a record as "reasoning-ready" only if its `narrative_factual` is ≥150 tokens and its `narrative_cause` is ≥50 tokens. The published reports comfortably exceed both thresholds. Adding them upgrades the existing `avall.mdb` rows for the same accidents from "marginal" to "reasoning-ready by construction."

### 1.3 Scope decisions (approved in brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Modes | Aviation only | Project is aviation-focused; NTSB Reports.aspx aviation listing is civil-only, no military. |
| Report types | All aviation types (AAR, AAB, SIR, SS, SRR) | User said "scrape completely". Type tagged in manifest; SS/SRR keep `accident_id=NULL` and serve as background corpus for Stage 3 priors. |
| Output formats | PDF + extracted TXT + manifest CSV | TXT is the primary product for downstream NLP; PDF kept on disk so extraction can be re-run with a different tool without re-scraping NTSB. |
| Linkage to `avall.mdb` | First-class `ntsb_accident_id` column in manifest | Lets these reports merge with existing structured rows in UASC Stage 0 instead of floating as a separate silo. |
| Source mechanism | Playwright headless Chromium | Reports.aspx is a SharePoint JS app; plain `requests` returns an empty shell. Playwright is bulletproof and adds the only known-reliable single-step approach. |
| Convention conformance | Match `scrape_bea.py` (resumable phases, polite delays, print-based progress, CLI flags) | Consistency with existing scrapers. |

### 1.4 Out of scope

- Surface-transport modes (highway, rail, marine, pipeline, hazmat).
- Military aviation (NTSB does not investigate military accidents).
- The investigative *docket* materials at `data.ntsb.gov/Docket/` (interviews, photos, factual reports). Different system, different volume, different design.
- OCR for scanned older reports. If `pdfplumber` returns near-empty text on a PDF, the manifest marks it `extraction_status=ocr_needed` and the row is left for a future OCR pass. No OCR is performed in this script.

## 2. Architecture

### 2.1 Four phases, each independently resumable

```
Phase 1  Enumerate          Playwright loads Reports.aspx, applies the Aviation
                            filter, paginates through every page, harvests each
                            row's metadata + PDF URL.
                            Output: data/NTSB_REPORTS/listing.json

Phase 2  Download           For each entry not already on disk, fetch the PDF
                            via plain `requests` (faster than browser downloads
                            and avoids browser-managed download dialogs).
                            Output: data/NTSB_REPORTS/pdf/<report_number>.pdf

Phase 3  Extract            Run pdfplumber over each PDF, joining text from all
                            pages. Skip extraction when the txt file already
                            exists and is newer than its PDF.
                            Output: data/NTSB_REPORTS/txt/<report_number>.txt

Phase 4  Manifest           Walk pdf/ + txt/ + listing.json, extract accident
                            IDs from text, and rewrite the manifest from current
                            disk state. Always rewritten so the manifest is
                            authoritative.
                            Output: data/NTSB_REPORTS/manifest.csv
```

Each phase is a separate function and can be run in isolation via `--phase N`. A normal `python scrape_ntsb_reports.py` invocation runs phases 1–4 in sequence, with each phase checking what already exists on disk before doing work. The script is safe to interrupt with Ctrl-C at any point and re-run; it picks up exactly where it left off. This matches the resume convention of `scrape_bea.py`.

### 2.2 Output directory layout

```
data/NTSB_REPORTS/
├── pdf/                       # downloaded PDFs (source-of-truth, kept on disk)
├── txt/                       # pdfplumber-extracted text (primary product)
├── listing.json               # Phase 1 cache: full listing harvested from Reports.aspx
├── manifest.csv               # Phase 4 manifest joining everything together
├── failed_downloads.log       # PDFs that hit final HTTP failure
├── failed_extractions.log     # PDFs where pdfplumber returned near-empty text
└── .playwright/               # local Chromium install, gitignored
```

The whole `/data/NTSB_REPORTS` directory should be added to `.gitignore`, matching the existing project convention for `/data/NTSB_ASRS`, `/data/TSB_CANADA`, and `/data/FAA_AIDS`. (The other data sources are also excluded wholesale rather than per-subdirectory.) If the user later decides to commit the extracted `txt/` files as a build artifact, that is a separate repo-policy decision and can be done with a more selective rule at that time.

### 2.3 Dependencies

Two new Python packages, added to the README install instructions:

```
pip install playwright pdfplumber
playwright install chromium
```

Playwright is used only in Phase 1. Phases 2–4 use only `requests`, `pdfplumber`, and the standard library. The Chromium install is one-time (~300 MB) and is isolated under `data/NTSB_REPORTS/.playwright/` via the `PLAYWRIGHT_BROWSERS_PATH` environment variable so it does not pollute global state.

## 3. Manifest schema

The manifest CSV is the single bridge from these reports back into the unified corpus. It is rewritten from disk on every Phase 4 run.

```
manifest.csv columns
├── report_number          e.g. "AAR-23-01"   primary key
├── report_type            {AAR, AAB, SIR, SS, SRR, OTHER}
├── title                  full title from the listing
├── publication_date       YYYY-MM-DD when NTSB published the report
├── accident_date          YYYY-MM-DD when the accident occurred (when known)
├── location               free-text location from the listing
├── ntsb_accident_id       e.g. "DCA19MA086"  the UASC join key
├── accident_id_source     {pdf_first_page, listing, inferred, missing}
├── pdf_url                original NTSB URL the PDF was fetched from
├── pdf_path               relative path to local PDF
├── txt_path               relative path to local extracted text
├── pdf_pages              page count
├── txt_chars              character count of extracted text
├── extraction_status      {ok, ocr_needed, failed}
└── scraped_at             ISO timestamp of when this row was last updated
```

Fifteen columns total. Every column earns its place: `accident_id_source` exists so Stage 0 of UASC can audit which records used the fallback inference path before trusting them as join keys; `extraction_status` exists so a future OCR pass can target only the rows that need it; `pdf_pages` and `txt_chars` exist so the richness-score computation in Stage 0 can be performed without re-opening the PDF.

## 4. Accident-ID extraction strategy

NTSB accident IDs follow a fixed pattern: **3 letters + 2 digits + 2 letters + 3 digits** (e.g. `DCA19MA086`, `ERA22FA123`, `WPR21LA047`). The regex used is:

```
\b[A-Z]{3}\d{2}[A-Z]{2}\d{3}\b
```

This regex is shared with the unit test file (Section 6.2) so any future regex tweak is exercised by the test set.

The ID is searched for in three sources, in decreasing order of reliability. The first source that yields a hit wins, and `accident_id_source` records which one:

1. **`pdf_first_page`** — extract text from the first 2 pages of the PDF and run the regex. Almost every AAR/AAB has the accident ID stamped on the cover or first body page. **Primary source.**
2. **`listing`** — the Reports.aspx listing entry sometimes embeds the accident ID in the title or detail row. **Secondary fallback.**
3. **`inferred`** — last-resort fuzzy match against `data/NTSB_ASRS/avall.mdb` (this project already requires `mdbtools`, see top-level README). Query `events` for rows with `ev_date == report.accident_date` AND `ev_city ~ report.location`. If exactly one row matches, use its `ev_id`. If zero or multiple match, give up and fall through to `missing`. The mdb query is wrapped in a try/except so the entire fallback is gracefully skipped if `mdbtools` is unavailable or `avall.mdb` is missing — the script never hard-depends on it. **Last-resort fallback only used when both other sources miss.**

For reports where no accident ID can be recovered (typically Safety Studies and multi-accident reports), `ntsb_accident_id` is left blank and `accident_id_source = missing`. The `report_type = SS` value makes the reason explicit. Stage 0 of UASC uses `ntsb_accident_id` as the deterministic join key into `avall.mdb`; rows with a blank ID become standalone narrative records rather than upgrades to existing rows.

## 5. Error handling and resumability

| Failure mode | Handling |
|---|---|
| Playwright cannot load Reports.aspx (Phase 1) | Log, save partial `listing.json`, exit non-zero. Re-run to retry. |
| Aviation filter button not found / page DOM changed (Phase 1) | Save a `phase1_dom_dump.html` next to `listing.json`, log the missing selector, exit non-zero. The dump lets a human or future Claude diagnose the DOM change without re-running Playwright. |
| PDF download HTTP error (Phase 2) | Retry up to 3 times with exponential backoff (1s, 4s, 16s). On final failure, append the URL to `failed_downloads.log` and continue. |
| 5 consecutive HTTP errors (Phase 2) | Treat as remote-side incident: print summary, save state, exit non-zero. Same guard as `scrape_bea.py`. |
| `pdfplumber` extraction returns < 500 chars (Phase 3) | Mark `extraction_status = ocr_needed`, write whatever text was returned, append to `failed_extractions.log`. Continue. |
| `pdfplumber` raises exception (Phase 3) | Mark `extraction_status = failed`, write empty txt file, append to `failed_extractions.log`. Continue. |
| Accident-ID regex misses on all three sources (Phase 4) | `ntsb_accident_id = ""`, `accident_id_source = missing`. Not an error — expected for SS/SRR. |

Polite delay between PDF downloads defaults to 1.5 seconds, configurable via `--delay` flag (matches `scrape_bea.py` default).

## 6. Testing strategy

This is a one-shot data acquisition script, not library code. Testing is end-to-end and lightweight rather than exhaustive unit tests.

### 6.1 Smoke test

```
python data/scrape_ntsb_reports.py --limit 5
```

The `--limit N` flag truncates the listing to the N most recent reports (sorted by `publication_date` descending) before Phase 2. Smoke test passes if:

1. All four phases complete without exception.
2. 5 PDFs land in `data/NTSB_REPORTS/pdf/`, all > 10 KB.
3. 5 txt files land in `data/NTSB_REPORTS/txt/`, all > 1000 chars.
4. `manifest.csv` has 5 data rows + header.
5. ≥ 4 of the 5 rows have a non-blank `ntsb_accident_id` (1 miss tolerated for noise).

### 6.2 Accident-ID regex unit test

`data/test_ntsb_reports_extract.py` — a small `pytest`-style file containing ~10 hand-curated text snippets (real NTSB cover pages, copied verbatim from public PDFs) and the expected accident IDs. Catches regex regressions without needing to re-scrape. Run with `pytest data/test_ntsb_reports_extract.py`. The regex itself is imported from `scrape_ntsb_reports.py` so any future tweak to the regex is automatically exercised by the test set.

### 6.3 Manual validation

After the first full run, manually compare 3 randomly-sampled txt files against their source PDFs. Look for systematic problems like 2-column layout being read in the wrong order, footnotes being interleaved with body text, or tables being mangled. Any systematic problem must be fixed before the script is declared done — fixing one record will fix the whole class.

## 7. CLI surface

```
python data/scrape_ntsb_reports.py                    # full run / resume from any state
python data/scrape_ntsb_reports.py --limit 5          # smoke test (Phase 2 onward limited to 5 most recent)
python data/scrape_ntsb_reports.py --delay 2          # custom delay between PDF downloads (seconds)
python data/scrape_ntsb_reports.py --phase 1          # run only Phase 1 (re-enumerate)
python data/scrape_ntsb_reports.py --phase 4          # rebuild manifest only (cheap, no network)
python data/scrape_ntsb_reports.py --force-extract    # re-extract all txt files even if up to date
```

Default invocation runs phases 1 → 4 in sequence, skipping work that has already been done.

## 8. Integration with ACE-Graph (UASC Stage 0)

When this script's output enters UASC Stage 0:

- Each `manifest.csv` row with a non-blank `ntsb_accident_id` joins to the corresponding `avall.mdb` row by `ev_id`. The `txt_path` content becomes that record's `narrative_factual` (or `narrative_analysis`, depending on report section structure — to be decided in Stage 1 brainstorming).
- The richness-score computation in Stage 0 §3.4 uses `txt_chars` directly for the `narrative_factual` length term, no PDF re-opening required.
- Rows with `accident_id_source = inferred` are flagged in `harmonization_flags` so the UASC provenance is honest about which join keys are deterministic vs. probabilistic.
- Rows with `report_type ∈ {SS, SRR}` and `ntsb_accident_id = ""` are kept in a sibling table `uasc_background_corpus` for use as Stage 3 LLM-prior seeds; they are not treated as accident records.

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| NTSB redesigns Reports.aspx and breaks the Playwright selectors | Phase 1 saves a `phase1_dom_dump.html` on selector failure (Section 5). The DOM dump lets a human or future Claude fix selectors without re-running Playwright on the broken page. |
| Some older reports are scanned PDFs with no embedded text | Marked `extraction_status = ocr_needed` in the manifest, listed in `failed_extractions.log`. A future OCR pass (out of scope for this script) targets only those rows. |
| Multi-column PDF layouts produce garbled text | Manual validation step (Section 6.3) catches systematic layout problems on the first run. Fix in `pdfplumber` extraction parameters before declaring done. |
| Accident-ID regex misses a real ID format variant | Unit test set (Section 6.2) covers the format. Add new variants to the test as they are discovered. |
| Repeated full re-runs hammer NTSB | Phase 2 skips PDFs already on disk. The script is genuinely incremental; only the first run is large. Polite delay defaults to 1.5s. |

## 10. Deliverables

1. `data/scrape_ntsb_reports.py` — the four-phase scraper.
2. `data/test_ntsb_reports_extract.py` — accident-ID regex tests.
3. README.md update — new "NTSB Published Reports" data-source section with install instructions and re-scrape command.
4. `.gitignore` update — add `/data/NTSB_REPORTS` (matches the existing `/data/NTSB_ASRS`, `/data/TSB_CANADA`, `/data/FAA_AIDS` convention).
5. First-run output: populated `data/NTSB_REPORTS/` with PDFs, txts, and manifest.
