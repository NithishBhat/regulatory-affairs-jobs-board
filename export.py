"""Build site/jobs.json from the job pipeline's daily raw files.

Reads job-pipeline/data/raw/*_raw.csv and data/sponsorship_companies.csv (read-only; never writes
there). Stateless: every run rebuilds from all raw files, so first_seen = earliest raw file with the job.

    py export.py [--pipeline PATH] [--max-age 45]
"""
import argparse
import csv
import glob
import html
import json
import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
PIPELINE = Path(os.environ.get("JOB_PIPELINE", r"C:\Users\nithi\OneDrive\Desktop\job-pipeline"))

# Employer-owned job boards only. Third-party boards (Indeed, LinkedIn, Glassdoor, ZipRecruiter,
# Himalayas, regulatoryjobs.com, association career centers) and sources switched off for legitimacy
# (UKG, SmartRecruiters, SuccessFactors XML) are never published.
EMPLOYER_SOURCES = {
    "workday", "greenhouse", "lever", "ashby", "icims", "oracle", "bamboohr", "paylocity", "workable",
    "jazzhr", "recruitee", "rippling", "adp", "eightfold", "jibe", "careerspage", "careersite",
    "sitemap", "successfactors_rss",
}

STATES = {
    "AL": "alabama", "AK": "alaska", "AZ": "arizona", "AR": "arkansas", "CA": "california",
    "CO": "colorado", "CT": "connecticut", "DE": "delaware", "DC": "district of columbia",
    "FL": "florida", "GA": "georgia", "HI": "hawaii", "ID": "idaho", "IL": "illinois", "IN": "indiana",
    "IA": "iowa", "KS": "kansas", "KY": "kentucky", "LA": "louisiana", "ME": "maine", "MD": "maryland",
    "MA": "massachusetts", "MI": "michigan", "MN": "minnesota", "MS": "mississippi", "MO": "missouri",
    "MT": "montana", "NE": "nebraska", "NV": "nevada", "NH": "new hampshire", "NJ": "new jersey",
    "NM": "new mexico", "NY": "new york", "NC": "north carolina", "ND": "north dakota", "OH": "ohio",
    "OK": "oklahoma", "OR": "oregon", "PA": "pennsylvania", "RI": "rhode island",
    "SC": "south carolina", "SD": "south dakota", "TN": "tennessee", "TX": "texas", "UT": "utah",
    "VT": "vermont", "VA": "virginia", "WA": "washington", "WV": "west virginia", "WI": "wisconsin",
    "WY": "wyoming", "PR": "puerto rico",
}
NON_US = re.compile(r"\b(?:canada|mexico|united kingdom|\buk\b|ireland|germany|france|india|china|japan|"
                    r"singapore|australia|netherlands|switzerland|belgium|spain|italy|brazil|poland|"
                    r"denmark|sweden|israel|korea|costa rica|philippines)\b", re.I)
CAP_EXEMPT = re.compile(r"universit|college|hospital|medical center|health system|children's|"
                        r"cancer center|institute|foundation|clinic\b", re.I)
YEARS = re.compile(r"(\d{1,2})\s*\+?\s*(?:(?:-|–|to)\s*\d{1,2}\s*\+?\s*)?years?(?:'|’)?\s*(?:of\s+)?"
                   r"(?:\w+\s+){0,5}?experience", re.I)
GENERIC = {"iso9001", "iso17025", "labeling", "postmarket", "intlreg", "nda"}
_SUFFIXES = {
    "inc", "incorporated", "llc", "lllp", "llp", "lp", "ltd", "limited", "corp", "corporation",
    "co", "company", "plc", "pllc", "pc", "pa", "sa", "ag", "gmbh", "bv", "nv", "the",
}


def normalize_company(name):
    """Same rule as job-pipeline src/common.py, so names match sponsorship_companies.csv."""
    s = (name or "").lower().replace("&", " and ")
    s = re.sub(r"\bd/?b/?a\b.*", "", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    words = [w for w in s.split() if w not in _SUFFIXES]
    while words and words[-1] == "and":
        words.pop()
    return " ".join(words)


def load_vocab():
    v = json.loads((SITE / "vocab.json").read_text(encoding="utf-8"))
    comp = lambda items: [(x["id"], re.compile(x["pattern"], re.I), x) for x in items]
    return {
        "families": comp(v["families"]),
        "seniority": comp(v["seniority"]),
        "terms": comp(v["terms"]),
        "software": re.compile(v["software_qa"], re.I),
    }


def load_sponsors():
    out = {}
    with open(PIPELINE / "data" / "sponsorship_companies.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            entry = (int(r["total_filings"] or 0), int(r["all_h1b_filings"] or 0))
            names = [r["normalized_name"]] + [normalize_company(n) for n in r["original_names"].split(";")]
            for n in names:
                if n and (n not in out or out[n][1] < entry[1]):
                    out[n] = entry
    return out


def sponsor_lookup(sponsors):
    """Exact normalized name, else the biggest filer whose name starts with it ('parexel' ->
    'parexel international')."""
    by_first = {}
    for n, v in sponsors.items():
        by_first.setdefault(n.split(" ")[0], []).append((n, v))

    def find(company):
        n = normalize_company(company)
        if n in sponsors:
            return sponsors[n]
        if len(n) < 5:
            return (0, 0)
        hits = [v for name, v in by_first.get(n.split(" ")[0], []) if name.startswith(n + " ")]
        return max(hits, key=lambda v: v[1]) if hits else (0, 0)
    return find


def read_raw():
    """{job_id: (row, first_seen, last_seen)} across all raw files, latest row wins."""
    jobs = {}
    for path in sorted(glob.glob(str(PIPELINE / "data" / "raw" / "*_raw.csv"))):
        run_day = Path(path).name[:10]
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                prev = jobs.get(r["job_id"])
                jobs[r["job_id"]] = (r, prev[1] if prev else run_day, run_day)
    return jobs


def parse_state(loc):
    for code in re.findall(r"(?:^|[\s,\-(/])([A-Z]{2})(?=$|[\s,\-)/])", loc or ""):
        if code in STATES:
            return code
    low = (loc or "").lower()
    for code, name in STATES.items():
        if re.search(rf"\b{name}\b", low) and not (name == "washington" and "washington, dc" in low):
            return code
    return ""


def first(pats, text):
    return next((i for i, p, _ in pats if p.search(text)), "")


def years_required(desc):
    vals = [int(m.group(1)) for m in YEARS.finditer(desc or "") if 0 < int(m.group(1)) <= 20]
    return min(vals) if vals else None


def classify(r, vocab):
    title, desc = r["title"] or "", r["description"] or ""
    family = first(vocab["families"], title)
    if not family:
        return None
    terms = [i for i, p, _ in vocab["terms"] if p.search(desc) or p.search(title)]
    # Needs a regulated-industry signal (FDA, GMP, ISO 13485, pharma...), not just generic quality terms
    # like CAPA or FMEA, which automotive / aerospace / energy QA jobs share.
    regulated = {t for t, _, x in vocab["terms"]
                 if t in terms and x["cat"] in ("standard", "submission", "industry") and t not in GENERIC}
    if family != "RA" and vocab["software"].search(title):
        return None
    if not regulated:
        return None
    return family, terms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-age", type=int, default=45, help="drop jobs posted (or first seen) longer ago")
    args = ap.parse_args()
    csv.field_size_limit(10**9)
    vocab, sponsor = load_vocab(), sponsor_lookup(load_sponsors())
    cutoff = (date.today() - timedelta(days=args.max_age)).isoformat()
    kept, seen_keys, skipped = [], set(), {"source": 0, "not_ra_qa": 0, "old": 0, "non_us": 0, "dup": 0}

    for job_id, (r, first_seen, last_seen) in sorted(read_raw().items(), key=lambda kv: kv[1][2], reverse=True):
        src = job_id.split(":")[0]
        if src not in EMPLOYER_SOURCES and not src.startswith("sitemap"):
            skipped["source"] += 1
            continue
        posted = r["date_posted"] if re.match(r"\d{4}-\d{2}-\d{2}$", r["date_posted"] or "") else first_seen
        if posted < cutoff:
            skipped["old"] += 1
            continue
        r["title"] = re.sub(r"^[^\w(]+", "", html.unescape(r["title"] or "").replace("Â", "").split(" | ")[0]).strip()
        loc = (r["location"] or "").strip()
        state = parse_state(loc)
        if not state and NON_US.search(loc):
            skipped["non_us"] += 1
            continue
        c = classify(r, vocab)
        if not c:
            skipped["not_ra_qa"] += 1
            continue
        company = html.unescape(r["company"] or "").replace("\xa0", " ").strip()
        key = (normalize_company(company), re.sub(r"\W+", " ", r["title"].lower()).strip(), state)
        if key in seen_keys:
            skipped["dup"] += 1
            continue
        seen_keys.add(key)
        family, terms = c
        ra_qa_filings, all_filings = sponsor(company)
        title = r["title"].strip()
        kept.append({
            "id": job_id,
            "title": title,
            "company": company,
            "location": loc,
            "state": state,
            "remote": r["remote"] == "yes" or bool(re.search(r"\bremote\b", loc, re.I)),
            "posted": posted,
            "url": r["job_url"],
            "family": family,
            "seniority": first(vocab["seniority"], title) or "mid",
            "years": years_required(r["description"]),
            "terms": terms,
            "h1b_ra_qa": ra_qa_filings,
            "h1b_all": all_filings,
            "cap_exempt": bool(CAP_EXEMPT.search(company)),
        })

    kept.sort(key=lambda j: j["posted"], reverse=True)
    out = {"generated": datetime.now().isoformat(timespec="minutes"), "count": len(kept), "jobs": kept}
    (SITE / "jobs.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    fams = {f: sum(j["family"] == f for j in kept) for f in ("RA", "QA", "QC", "DOC")}
    print(f"{len(kept)} jobs -> site/jobs.json {fams}; skipped {skipped}")


if __name__ == "__main__":
    main()
