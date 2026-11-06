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


def test_restrictions():
    cases = {
        "citizen": ["Must be a U.S. citizen.", "US Citizenship Required: Yes", "U.S. citizenship is required.",
                    "Open to US citizens only", "Green card holders only.", "Permanent residents only",
                    "Must be a US citizen or permanent resident."],
        "clearance": ["Active TS/SCI clearance with polygraph", "Ability to obtain and maintain a SECRET security clearance",
                      "Security Clearance Required: (TS/SCI + Poly)"],
        "export": ["This position requires U.S. person status under ITAR regulations.",
                   'successful applicants must be a "US Person" (US Citizens, US Nationals)',
                   "This position may require exposure to export-controlled information.",
                   "Candidates must meet export control requirements.",
                   "Must be a U.S. Person (U.S. Citizen, or U.S. Permanent Resident)"],
        "no_sponsor": ["This role does not offer sponsorship for work authorization.",
                       "Must be able to work in the United States without sponsorship.",
                       "Is Sponsorship Available? No", "We are unable to sponsor or take over sponsorship of a visa."],
    }
    for kind, texts in cases.items():
        for t in texts:
            assert kind in export.restrictions(t), (kind, t)
    for t in ["Work Authorization: US Citizen,Green Card,OPT-EAD,CPT,H-1B,H4-EAD",
              "US Citizenship Required: No",
              "Manage export control classifications, including ECCN and USML.",
              "Please be aware that certain positions may require the applicant to be a U.S. citizen or national.",
              "some roles may be subject to U.S. export controls for nationals of embargoed countries",
              "Must be a U.S. citizen, U.S. national, legal permanent resident, asylee, refugee",
              "We sponsor H-1B visas for this role.", "Experience with customs clearance."]:
        assert not set(export.restrictions(t)) & {"citizen", "clearance", "no_sponsor"}, t


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
