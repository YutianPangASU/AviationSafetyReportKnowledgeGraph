"""
Download all FAA AIDS (Accident/Incident Data System) files from ASIAS.
https://www.asias.faa.gov/apex/f?p=100:189:::NO:::

Data is TAB-delimited text in zip files, split by time period.
  A* files = accident/incident records
  E* files = edited remarks/narratives
  acftser  = aircraft make/model/series lookup
  airport  = airport/location lookup

Output: data/FAA_AIDS/
"""

import requests
import re
import os
import sys

BASE_URL = "https://www.asias.faa.gov/apex/"
PAGE_URL = BASE_URL + "f?p=100:189:::NO:::"
OUTPUT_DIR = "data/FAA_AIDS"


def scrape_and_download():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    session = requests.Session()

    # Get the page to extract session-bound download URLs
    print("Fetching AIDS download page...", flush=True)
    resp = session.get(PAGE_URL)
    resp.raise_for_status()
    html = resp.text

    # Extract all download links: both metadata and data files
    # Pattern for metadata files (doc, txt)
    meta_links = re.findall(
        r'<td[^>]*headers="FILENAME">([\w\.\-]+)</td>.*?href="(apex_util\.get_blob[^"]+)"',
        html, re.DOTALL
    )

    print(f"Found {len(meta_links)} files to download\n", flush=True)

    for filename, url in meta_links:
        filepath = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(filepath):
            print(f"  SKIP (exists): {filename}", flush=True)
            continue

        full_url = BASE_URL + url.replace("&amp;", "&")
        print(f"  Downloading: {filename}...", end=" ", flush=True)
        try:
            resp = session.get(full_url)
            resp.raise_for_status()
            with open(filepath, 'wb') as f:
                f.write(resp.content)
            size_kb = len(resp.content) / 1024
            print(f"{size_kb:.0f}KB", flush=True)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)

    # Unzip all zip files
    import zipfile
    print("\nExtracting zip files...", flush=True)
    for fname in os.listdir(OUTPUT_DIR):
        if fname.endswith('.zip'):
            zpath = os.path.join(OUTPUT_DIR, fname)
            try:
                with zipfile.ZipFile(zpath, 'r') as z:
                    z.extractall(OUTPUT_DIR)
                print(f"  Extracted: {fname}", flush=True)
            except Exception as e:
                print(f"  ERROR extracting {fname}: {e}", flush=True)

    print(f"\nDone! Files saved to {OUTPUT_DIR}/", flush=True)
    print("\nDirectory contents:", flush=True)
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"  {f:30s} {size/1024:,.0f} KB", flush=True)


if __name__ == '__main__':
    scrape_and_download()
