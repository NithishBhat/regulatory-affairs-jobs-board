"""Tests for export.py's location and job filters (no pipeline data needed).

    py tools/test/test_export.py
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("export", ROOT / "export.py")
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def non_us(loc, title="QA Specialist"):
    code = export.COUNTRY_CODE.search(loc)
    return bool((not export.parse_state(loc) and export.NON_US.search(loc)) or export.GENDER_TAG.search(title)
                or (code and code.group(1) != "US"))


def test_states():
    assert export.parse_state("Piscataway, NJ, United States, US") == "NJ"
    assert export.parse_state("Albuquerque, New Mexico") == "NM"
    assert export.parse_state("Remote (United States)") == ""
    assert export.parse_state("Washington, DC") != "WA"


def test_non_us():
    for loc, title in [("Riyadh plant, Al Riyadh, SA", "QA"), ("Nehren, UNAVAILABLE, DE", "RA (m/w/d)"),
                       ("Remote, Belgio", "QA"), ("Ottawa, ON, CA", "QA"), ("Paris", "Ingénieur Qualité (H/F)"),
                       ("Bangalore, India", "QA"), ("Toronto, Canada", "QA")]:
        assert non_us(loc, title), loc
    for loc in ["Piscataway, NJ, United States, US", "Post Falls, North America, US", "Albuquerque, New Mexico",
                "Dover, DE", "Wilmington, NC, USA", "Remote (United States)", "", "Indianapolis, IN"]:
        assert not non_us(loc), loc


def test_years_required():
    assert export.years_required("Requires 5+ years of regulatory experience") == 5
    assert export.years_required("3-5 years experience in GMP") == 3
    assert export.years_required("No experience needed") is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all passed")
