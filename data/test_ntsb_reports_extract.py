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
