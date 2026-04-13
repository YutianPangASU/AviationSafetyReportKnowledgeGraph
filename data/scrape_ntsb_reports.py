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

# ----- Manifest schema (15 columns, see spec section 3) -----
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


def extract_accident_id_from_text(text):
    """Return the first NTSB accident ID found in `text`, or None.

    NTSB accident IDs match the pattern: 3 uppercase letters + 2 digits +
    2 uppercase letters + 3 digits, e.g. DCA19MA086, ERA20FA123.
    """
    if not text:
        return None
    match = ACCIDENT_ID_RE.search(text)
    return match.group(0) if match else None


def enumerate_listing(force=False):
    """Phase 1: Use Playwright to load Reports.aspx, click the Aviation
    filter button, and paginate through every page harvesting each report
    block (title, location, dates, report number, accident page link, PDF URL).

    Caches result to LISTING_FILE. Skips work if cache exists and force=False.
    Returns the list of listing entries.
    """
    if os.path.exists(LISTING_FILE) and not force:
        with open(LISTING_FILE, "r", encoding="utf-8") as f:
            entries = json.load(f)
        print(f"[Phase 1] Loaded {len(entries)} entries from cache: {LISTING_FILE}", flush=True)
        return entries

    # Lazy import so the script can run later phases without Playwright.
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
            # SharePoint defers some rendering past 'networkidle'; give it a beat.
            page.wait_for_timeout(3000)

            # Wait for at least one PDF anchor to appear (means the listing rendered).
            try:
                page.wait_for_selector("div.block div.download a[href$='.pdf']", timeout=30000)
            except PwTimeout:
                _dump_dom_and_fail(page, "initial listing did not render")

            # Click the Aviation filter button.
            try:
                aviation_btn = page.get_by_role("button", name=re.compile(r"^Aviation$", re.I)).first
                if aviation_btn.count() == 0:
                    _dump_dom_and_fail(page, "Aviation filter button not found")
                aviation_btn.click()
                page.wait_for_timeout(2000)
                # Confirm filter took effect: AIR/AAR/AAB/etc. PDFs only.
                page.wait_for_selector("div.block div.download a[href$='.pdf']", timeout=15000)
            except PwTimeout:
                _dump_dom_and_fail(page, "Aviation filter did not render results")
            except Exception as e:
                print(f"[Phase 1] Aviation filter click failed: {e}", flush=True)
                _dump_dom_and_fail(page, "aviation filter not clickable")

            # Paginate and harvest
            page_num = 1
            seen_first_pdf = None
            while True:
                page_entries = _harvest_current_page(page)
                if not page_entries:
                    if page_num == 1:
                        _dump_dom_and_fail(page, "no entries found on page 1")
                    print(f"[Phase 1] Page {page_num}: empty, stopping", flush=True)
                    break

                # Guard: detect if Next click silently failed and we got the same page back.
                first_pdf = page_entries[0]["pdf_url"]
                if first_pdf == seen_first_pdf:
                    print(f"[Phase 1] Page {page_num}: same as previous, stopping", flush=True)
                    break
                seen_first_pdf = first_pdf

                entries.extend(page_entries)
                print(
                    f"[Phase 1] Page {page_num}: {len(page_entries)} entries (total {len(entries)})",
                    flush=True,
                )

                # Check the Next button: <li id="NextLinkItem"> with class "disabled" when at end.
                next_li = page.locator("li#NextLinkItem")
                if next_li.count() == 0:
                    print("[Phase 1] No NextLinkItem element, stopping", flush=True)
                    break
                cls = next_li.first.get_attribute("class") or ""
                if "disabled" in cls:
                    print(f"[Phase 1] Reached last page ({page_num})", flush=True)
                    break

                try:
                    page.locator("li#NextLinkItem a").first.click()
                    page.wait_for_timeout(1500)
                except Exception as e:
                    print(f"[Phase 1] Next click failed: {e}", flush=True)
                    break
                page_num += 1
                time.sleep(0.5)

        finally:
            browser.close()

    # Deduplicate by pdf_url (defensive; the real source is unique pages)
    seen = set()
    deduped = []
    for e in entries:
        if e["pdf_url"] in seen:
            continue
        seen.add(e["pdf_url"])
        deduped.append(e)

    print(f"[Phase 1] Done. {len(deduped)} unique entries.", flush=True)
    with open(LISTING_FILE, "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2)
    print(f"[Phase 1] Saved to {LISTING_FILE}", flush=True)
    return deduped


def _harvest_current_page(page):
    """Extract every report block from the currently-rendered listing page.

    Block structure (from live DOM inspection):
      div.block
        div.desc
          p > a[href="/investigations/Pages/<ACCIDENT_ID>.aspx"]   -> title text
          p.location > span.title                                  -> location text
          p.data      -> "Accident Date: M/D/YYYY" + "Report Date: M/D/YYYY"
          p.report > span.title -> "Report Number: " + value
        div.download
          p > a[href="/investigations/AccidentReports/Reports/<file>.pdf"]
    """
    entries = []
    blocks = page.locator("div.block").all()
    for block in blocks:
        try:
            # PDF link
            pdf_loc = block.locator("div.download a[href$='.pdf']")
            if pdf_loc.count() == 0:
                continue
            pdf_href = pdf_loc.first.get_attribute("href") or ""
            if pdf_href.startswith("/"):
                pdf_url = "https://www.ntsb.gov" + pdf_href
            elif pdf_href.startswith("http"):
                pdf_url = pdf_href
            else:
                continue

            # Title and accident-page link (which contains the NTSB accident ID in its URL)
            title = ""
            desc_href = ""
            desc_a = block.locator("div.desc > p:first-child > a").first
            if desc_a.count() > 0:
                title = (desc_a.text_content() or "").strip()
                desc_href = desc_a.get_attribute("href") or ""

            # Accident ID from desc_href like /investigations/Pages/DCA25MA108.aspx
            ntsb_accident_id = ""
            m = ACCIDENT_ID_RE.search(desc_href)
            if m:
                ntsb_accident_id = m.group(0)

            # Location
            location = ""
            loc_loc = block.locator("p.location span.title").first
            if loc_loc.count() > 0:
                location = (loc_loc.text_content() or "").strip()
                if location.lower().startswith("no location"):
                    location = ""

            # Dates from p.data
            accident_date = ""
            publication_date = ""
            data_loc = block.locator("p.data").first
            if data_loc.count() > 0:
                data_text = (data_loc.text_content() or "").strip()
                m_acc = re.search(r"Accident Date:\s*([0-9NA/]+)", data_text)
                if m_acc and m_acc.group(1) not in ("N/A", ""):
                    accident_date = _normalize_date(m_acc.group(1))
                m_pub = re.search(r"Report Date:\s*([0-9NA/]+)", data_text)
                if m_pub and m_pub.group(1) not in ("N/A", ""):
                    publication_date = _normalize_date(m_pub.group(1))

            # Report number from p.report
            report_number = ""
            rep_loc = block.locator("p.report").first
            if rep_loc.count() > 0:
                rep_text = (rep_loc.text_content() or "").strip()
                m_rn = re.search(r"Report Number:\s*([A-Z0-9\-]+)", rep_text)
                if m_rn:
                    report_number = m_rn.group(1)
            if not report_number:
                report_number = _infer_report_number_from_url(pdf_url)

            report_type = _infer_report_type(report_number)

            entries.append({
                "report_number": report_number,
                "report_type": report_type,
                "title": title,
                "publication_date": publication_date,
                "accident_date": accident_date,
                "location": location,
                "ntsb_accident_id": ntsb_accident_id,
                "pdf_url": pdf_url,
            })
        except Exception as e:
            print(f"[Phase 1]   block error: {e}", flush=True)
            continue
    return entries


def _normalize_date(s):
    """Convert M/D/YYYY or YYYY-MM-DD to YYYY-MM-DD. Returns '' on failure."""
    if not s:
        return ""
    s = s.strip()
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
    if m:
        return s
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", s)
    if m:
        mm, dd, yy = m.groups()
        return f"{yy}-{int(mm):02d}-{int(dd):02d}"
    return ""


def _infer_report_number_from_url(url):
    """e.g. .../Reports/AAR2301.pdf -> AAR-23-01, .../AIR-24-07.pdf -> AIR-24-07"""
    base = os.path.splitext(os.path.basename(url))[0]
    if "-" in base:
        return base
    m = re.match(r"([A-Z]+)(\d{2})(\d{2,3})$", base)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return base


def _infer_report_type(report_number):
    """Classify the report by its number prefix.
    AAR=Aircraft Accident Report, AAB=Aircraft Accident Brief,
    SIR=Special Investigation Report, SS=Safety Study, SRR=Safety Recommendation Report,
    AIR=Aircraft Investigation Report (newer naming), ASR=Aviation Safety Report.
    """
    rn_upper = (report_number or "").upper()
    # Order matters: prefer longer prefixes first
    for code in ("AAR", "AAB", "SIR", "AIR", "ASR", "SRR", "SS", "SR"):
        if rn_upper.startswith(code):
            if code == "SR":
                return "SRR"
            return code
    return "OTHER"


def _dump_dom_and_fail(page, reason):
    """Save the current page HTML to PHASE1_DOM_DUMP and raise."""
    try:
        with open(PHASE1_DOM_DUMP, "w", encoding="utf-8") as f:
            f.write(page.content())
        print(f"[Phase 1] DOM dumped to {PHASE1_DOM_DUMP} for inspection.", flush=True)
    except Exception:
        pass
    raise RuntimeError(f"Phase 1 failed: {reason}")


def download_pdfs(entries, delay=1.5, limit=None):
    """Phase 2: Download each entry's PDF to PDF_DIR. Skips files already on
    disk. Retries 3x on HTTP error. Aborts after 5 consecutive failures.

    Returns the count of newly downloaded PDFs.
    """
    if limit is not None:
        sorted_entries = sorted(
            entries,
            key=lambda e: e.get("publication_date", "") or "",
            reverse=True,
        )
        entries = sorted_entries[:limit]
        print(
            f"[Phase 2] --limit {limit}: restricted to {len(entries)} most recent entries",
            flush=True,
        )

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    failed = []
    consecutive_errors = 0
    new_count = 0

    for i, entry in enumerate(entries):
        report_number = entry["report_number"]
        # Sanitize for filesystem (no slashes)
        safe_name = report_number.replace("/", "_")
        pdf_path = os.path.join(PDF_DIR, f"{safe_name}.pdf")
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
                print(
                    f"[Phase 2] [{i+1}/{len(entries)}] {report_number:18s} {size_kb:7.0f} KB",
                    flush=True,
                )
                success = True
                consecutive_errors = 0
                new_count += 1
                break
            except Exception as e:
                wait = 1 * (4 ** attempt)
                print(
                    f"[Phase 2]   attempt {attempt+1} for {report_number} failed: {e}; sleeping {wait}s",
                    flush=True,
                )
                time.sleep(wait)

        if not success:
            failed.append(url)
            consecutive_errors += 1
            with open(FAILED_DOWNLOADS_LOG, "a", encoding="utf-8") as f:
                f.write(
                    f"{datetime.now(timezone.utc).isoformat()}\t{report_number}\t{url}\n"
                )
            if consecutive_errors >= 5:
                print(
                    "[Phase 2] 5 consecutive failures, aborting. Re-run to resume.",
                    flush=True,
                )
                break

        time.sleep(delay)

    print(
        f"[Phase 2] Done. {new_count} new PDFs downloaded. {len(failed)} failed.",
        flush=True,
    )
    if failed:
        print(f"[Phase 2] Failed URLs logged to {FAILED_DOWNLOADS_LOG}", flush=True)
    return new_count


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
                    f.write(
                        f"{datetime.now(timezone.utc).isoformat()}\t{report_number}\tneeds_ocr\t{len(full_text)} chars\n"
                    )
                print(
                    f"[Phase 3] [{i+1}/{len(pdfs)}] {report_number}: only {len(full_text)} chars (ocr_needed)",
                    flush=True,
                )
            else:
                statuses[report_number] = "ok"
                print(
                    f"[Phase 3] [{i+1}/{len(pdfs)}] {report_number}: {len(full_text)} chars",
                    flush=True,
                )

        except Exception as e:
            print(
                f"[Phase 3] [{i+1}/{len(pdfs)}] {report_number}: extraction failed: {e}",
                flush=True,
            )
            statuses[report_number] = "failed"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("")
            with open(FAILED_EXTRACTIONS_LOG, "a", encoding="utf-8") as f:
                f.write(
                    f"{datetime.now(timezone.utc).isoformat()}\t{report_number}\tfailed\t{e}\n"
                )

    print("[Phase 3] Done.", flush=True)
    return statuses


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


def extract_accident_id_from_mdb(entry):
    """Last-resort: query avall.mdb for an event matching this report's
    accident_date and location. Returns ev_id or None. Gracefully no-ops if
    mdbtools or avall.mdb is unavailable.
    """
    mdb_path = "data/NTSB_ASRS/avall.mdb"
    if not os.path.exists(mdb_path):
        return None
    accident_date = entry.get("accident_date") or ""
    location = entry.get("location") or ""
    if not accident_date or not location:
        return None
    try:
        import subprocess
        import io
        result = subprocess.run(
            ["mdb-export", mdb_path, "events"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            return None
        reader = csv.DictReader(io.StringIO(result.stdout))
        candidates = []
        loc_lower = location.lower()
        for row in reader:
            ev_date = (row.get("ev_date") or "")[:10]
            if ev_date != accident_date:
                continue
            ev_city = (row.get("ev_city") or "").lower()
            ev_state = (row.get("ev_state") or "").lower()
            if (ev_city and ev_city in loc_lower) or (ev_state and ev_state in loc_lower):
                candidates.append(row.get("ev_id"))
        if len(candidates) == 1:
            return candidates[0]
        return None
    except Exception:
        return None


def build_manifest():
    """Phase 4: Walk pdf/ + txt/ + listing.json, extract accident IDs, and
    rewrite manifest.csv from current disk state.

    Three-tier accident ID resolution:
      1. listing       (already populated by Phase 1 from desc link href)
      2. pdf_first_page (regex over first 2 pages of extracted PDF text)
      3. inferred      (avall.mdb fuzzy match by date + location, optional)
    """
    if not os.path.exists(LISTING_FILE):
        print("[Phase 4] No listing.json yet. Run Phase 1 first.", flush=True)
        return
    with open(LISTING_FILE, "r", encoding="utf-8") as f:
        entries = json.load(f)
    by_report = {e["report_number"]: e for e in entries}

    rows = []
    pdfs = sorted(f for f in os.listdir(PDF_DIR) if f.endswith(".pdf"))
    print(f"[Phase 4] Building manifest for {len(pdfs)} PDFs...", flush=True)

    import pdfplumber  # for page count

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
            "ntsb_accident_id": "",
            "pdf_url": "",
        })

        # Three-tier accident-ID resolution
        acc_id = entry.get("ntsb_accident_id") or ""
        acc_src = "listing" if acc_id else None
        if not acc_id:
            acc_id = extract_accident_id_from_pdf_first_pages(pdf_path) or ""
            if acc_id:
                acc_src = "pdf_first_page"
        if not acc_id:
            mdb_id = extract_accident_id_from_mdb(entry) or ""
            if mdb_id:
                acc_id = mdb_id
                acc_src = "inferred"
        if not acc_id:
            acc_src = "missing"

        # PDF page count
        pdf_pages = 0
        try:
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
            "ntsb_accident_id": acc_id,
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
    print(
        f"[Phase 4] Done. {len(rows)} rows, {n_with_id} with accident ID. Saved to {MANIFEST_FILE}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Phase 2 only: download just the N most recent reports (smoke test)",
    )
    parser.add_argument(
        "--delay", type=float, default=1.5,
        help="Delay between PDF downloads in seconds (default: 1.5)",
    )
    parser.add_argument(
        "--phase", type=int, choices=[1, 2, 3, 4], default=None,
        help="Run only the specified phase (1=enumerate, 2=download, 3=extract, 4=manifest)",
    )
    parser.add_argument(
        "--force-extract", action="store_true",
        help="Phase 3: re-extract all txt files even if up to date",
    )
    parser.add_argument(
        "--force-enumerate", action="store_true",
        help="Phase 1: re-enumerate even if listing.json exists",
    )
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
                print(
                    "[main] Cannot run Phase 2: listing.json missing. Run Phase 1 first.",
                    flush=True,
                )
                sys.exit(1)
            with open(LISTING_FILE, "r", encoding="utf-8") as f:
                entries = json.load(f)
        download_pdfs(entries, delay=args.delay, limit=args.limit)
    if 3 in phases_to_run:
        extract_text(force=args.force_extract)
    if 4 in phases_to_run:
        build_manifest()

    print("[main] All requested phases complete.", flush=True)


if __name__ == "__main__":
    main()
