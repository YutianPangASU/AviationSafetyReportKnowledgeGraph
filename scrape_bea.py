"""
Scrape all notified events from the BEA (Bureau d'Enquetes et d'Analyses)
https://bea.aero/en/investigation-reports/notified-events/

Two-phase scrape:
  Phase 1: Crawl all list pages to get event URLs (cached to urls.txt)
  Phase 2: Fetch each detail page for full info (resumes from existing CSV)

Output: data/BEA/bea_notified_events.csv

Usage:
  python scrape_bea.py            # full scrape or resume
  python scrape_bea.py --delay 2  # custom delay (default 1.5s)
"""

import requests
import re
import csv
import time
import os
import argparse
from html import unescape

BASE_URL = "https://bea.aero/en/investigation-reports/notified-events/"
ITEMS_PER_PAGE = 10
OUTPUT_DIR = "data/BEA"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "bea_notified_events.csv")
URLS_FILE = os.path.join(OUTPUT_DIR, "urls.txt")

requests.packages.urllib3.disable_warnings()

FIELDS = [
    "file_number", "title", "summary", "date", "location",
    "state_of_occurrence", "occurrence_class", "human_consequences",
    "aircraft_category", "manufacturer_model", "registration",
    "state_of_registry", "operator", "operation_type",
    "flight_phase", "departure", "destination",
    "responsible_entity", "detail_url",
]


def clean_text(text):
    if text is None:
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    return unescape(re.sub(r'\s+', ' ', text)).strip()


def get_total_events(html):
    match = re.search(r'<strong>(\d[\d,\s]*)\s*notified events', html)
    if match:
        return int(match.group(1).replace(',', '').replace(' ', ''))
    return None


def parse_list_page(html):
    urls = []
    parts = html.split('<article class="search-entry">')
    for part in parts[1:]:
        end = part.find('</article>')
        if end == -1:
            continue
        block = part[:end]
        match = re.search(r'search-entry__title">\s*<a\s+href="([^"]*)"', block, re.DOTALL)
        if match:
            urls.append("https://bea.aero" + match.group(1).strip())
    return urls


def parse_detail_page(html):
    record = {f: "" for f in FIELDS}

    m = re.search(r'inv__intro">\s*.*?<h1>(.*?)</h1>', html, re.DOTALL)
    if m:
        record['title'] = clean_text(m.group(1))

    m = re.search(r'inv__bea-disclaimer-small">(.*?)</span>', html, re.DOTALL)
    short_desc = clean_text(m.group(1)) if m else ""

    m = re.search(r'<section class="inv__section" id="resume">\s*(.*?)\s*</section>', html, re.DOTALL)
    if m:
        record['summary'] = clean_text(m.group(1))
    else:
        record['summary'] = short_desc

    start = html.find('General information')
    if start < 0:
        start = 0
    end = html.find('TYPO3SEARCH_end')
    if end < 0:
        end = len(html)
    chunk = html[start:end]

    lines = re.sub(r'<[^>]+>', '\n', chunk)
    lines = [l.strip() for l in lines.split('\n') if l.strip()]

    label_map = {
        'local date': 'date',
        'state or area of occurrence': 'state_of_occurrence',
        'location': 'location',
        'human consequences': 'human_consequences',
        'occurrence class': 'occurrence_class',
        'file number': 'file_number',
        'aircraft category': 'aircraft_category',
        'manufacturer / model': 'manufacturer_model',
        'aircraft registration': 'registration',
        'state of registry': 'state_of_registry',
        'operator': 'operator',
        'operation type': 'operation_type',
        'flight phase': 'flight_phase',
        'last departure point': 'departure',
        'planned destination': 'destination',
        'responsible entity': 'responsible_entity',
        'aircraft consequences': 'aircraft_consequences',
    }

    i = 0
    while i < len(lines) - 1:
        key = lines[i].lower().strip()
        if key in label_map:
            field = label_map[key]
            if field in record:
                record[field] = lines[i + 1].strip()
            i += 2
        else:
            i += 1

    return record


def load_existing_records():
    records = []
    fetched_urls = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
                if row.get('detail_url'):
                    fetched_urls.add(row['detail_url'])
    return records, fetched_urls


def save_csv(records):
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def collect_urls(session, delay):
    """Phase 1: collect all detail URLs. Caches to urls.txt."""
    if os.path.exists(URLS_FILE):
        with open(URLS_FILE, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(urls)} URLs from cache ({URLS_FILE})", flush=True)
        return urls

    print("=== Phase 1: Collecting event URLs from list pages ===", flush=True)
    resp = session.get(BASE_URL, timeout=30)
    resp.raise_for_status()
    total = get_total_events(resp.text)
    print(f"Total notified events: {total}", flush=True)
    total_pages = (total // ITEMS_PER_PAGE) + (1 if total % ITEMS_PER_PAGE else 0)
    print(f"Total pages: {total_pages}", flush=True)

    all_urls = parse_list_page(resp.text)
    print(f"  Page 1: {len(all_urls)} URLs", flush=True)

    for page in range(2, total_pages + 1):
        url = f"{BASE_URL}?tx_news_pi1%5Bpage%5D={page}"
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            urls = parse_list_page(resp.text)
            all_urls.extend(urls)
            if page % 50 == 0 or page == total_pages:
                print(f"  Page {page}/{total_pages}: {len(all_urls)} URLs collected", flush=True)
        except Exception as e:
            print(f"  ERROR on page {page}: {e}", flush=True)
        time.sleep(delay)

    # Cache to disk
    with open(URLS_FILE, 'w') as f:
        for u in all_urls:
            f.write(u + '\n')
    print(f"Saved {len(all_urls)} URLs to {URLS_FILE}", flush=True)
    return all_urls


def scrape_all(delay=1.5):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    session = requests.Session()
    session.verify = False
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                       '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })

    # Load existing progress
    all_records, fetched_urls = load_existing_records()
    if fetched_urls:
        print(f"Resuming: {len(fetched_urls)} records already scraped", flush=True)

    # Phase 1: get URLs (from cache or by scraping)
    all_urls = collect_urls(session, delay)

    # Filter out already-fetched
    remaining_urls = [u for u in all_urls if u not in fetched_urls]
    print(f"Remaining to fetch: {len(remaining_urls)}", flush=True)

    if not remaining_urls:
        print("All done! Nothing to fetch.", flush=True)
        return

    # Phase 2: fetch detail pages
    print(f"\n=== Phase 2: Fetching detail pages (delay={delay}s) ===", flush=True)
    consecutive_errors = 0
    for i, detail_url in enumerate(remaining_urls):
        try:
            resp = session.get(detail_url, timeout=30)
            resp.raise_for_status()
            record = parse_detail_page(resp.text)
            record['detail_url'] = detail_url
            all_records.append(record)
            consecutive_errors = 0
        except Exception as e:
            print(f"  ERROR on {detail_url}: {e}", flush=True)
            all_records.append({'detail_url': detail_url})
            consecutive_errors += 1
            if consecutive_errors >= 5:
                print("  Too many consecutive errors, saving and exiting.", flush=True)
                save_csv(all_records)
                print(f"  Saved {len(all_records)} records. Re-run to resume.", flush=True)
                return

        if (i + 1) % 100 == 0 or (i + 1) == len(remaining_urls):
            print(f"  {i+1}/{len(remaining_urls)} new pages ({len(all_records)} total)", flush=True)
            save_csv(all_records)

        time.sleep(delay)

    save_csv(all_records)
    print(f"\nDone! Saved {len(all_records)} events to {OUTPUT_FILE}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--delay', type=float, default=1.5,
                        help='Delay between requests in seconds (default: 1.5)')
    args = parser.parse_args()
    scrape_all(delay=args.delay)
