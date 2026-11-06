"""Build site/jobs.json from the job pipeline's daily raw files.

Reads job-pipeline/data/raw/*_raw.csv and data/sponsorship_companies.csv (read-only; never writes
there). Every run rebuilds from all raw files, so first_seen = earliest raw file with the job. Closed
postings are dropped: ids in the pipeline's data/closed_jobs.csv, plus a daily link check through the
pipeline's src/liveness.py, cached in cache/liveness.csv.

    py export.py [--max-age 45] [--no-check]
"""
import argparse
import csv
import glob
import html
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
CACHE = ROOT / "cache" / "liveness.csv"
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
                    r"denmark|sweden|israel|korea|costa rica|philippines|saudi|riyadh|uae|dubai|abu dhabi|qatar|"
                    r"egypt|turkey|south africa|nigeria|kenya|pakistan|bangladesh|vietnam|thailand|malaysia|"
                    r"indonesia|taiwan|hong kong|new zealand|argentina|chile|colombia|peru|portugal|austria|"
                    r"czech|hungary|romania|greece|finland|norway|slovakia|slovenia|croatia|serbia|ukraine|"
                    r"belgio|belgique|deutschland|germania|alemania|allemagne|svizzera|schweiz|suisse|"
                    r"nederland|espa[ñn]a|italia|polska|m[ée]xico|brasil)\b", re.I)
# Who can apply. Requirement wording only: "Work Authorization: US Citizen, Green Card, OPT-EAD, H-1B" (open
# to visa holders), "US Citizenship Required: No", export-control duties in a trade-compliance job, and
# "certain positions may require…" boilerplate are not restrictions.
_US = r"u\.?\s?s\.?|united\s+states|american"
_Q = r"[\"“”']?"
RESTRICT = {
    "citizen": re.compile(
        rf"(?:must|required\s+to|need\s+to|have\s+to)\s+be\s+(?:an?\s+)?(?:{_US})\s+citizens?(?![\s,]*(?:or\s+|and\s+)?(?:an?\s+)?(?:(?:{_US})\s+)?(?:national|lawful|legal|permanent|green))|"
        rf"(?:{_US})\s+citizenship\s+(?:is\s+)?(?:required|mandatory|needed)(?!\s*:?\s*no\b)|"
        rf"requires?\s+(?:\w+\s+){{0,2}}(?:{_US})\s+citizenship|"
        rf"(?:open|available|limited|restricted)\s+(?:only\s+)?to\s+(?:{_US})\s+citizens|"
        rf"only\s+(?:{_US})\s+citizens|(?:{_US})\s+citizens\s+only|"
        # green card / permanent residents only
        r"green\s*card\s+holders?\s+only|(?:lawful\s+|legal\s+)?permanent\s+residents?\s+only|"
        rf"(?:{_US})\s+citizens?\s+(?:and|or)\s+(?:lawful\s+|legal\s+)?(?:permanent\s+residents?|green\s*card\s+holders?)\s+only|"
        rf"must\s+be\s+(?:an?\s+)?(?:{_US})\s+citizen\s+or\s+(?:a\s+)?(?:lawful\s+|legal\s+)?(?:permanent\s+resident|green\s*card\s+holder)", re.I),
    "clearance": re.compile(
        r"security\s+clearance|secret\s+clearance|\bts\s*/\s*sci\b|top\s+secret|public\s+trust\s+clearance|"
        r"(?:dod|government|federal)\s+clearance|clearance\s+(?:is\s+)?required", re.I),
    "export": re.compile(
        rf"(?:{_US})\s+person{_Q}\s+status|must\s+be\s+(?:an?\s+)?{_Q}(?:{_US})\s+person|"
        rf"requires?\s+(?:\w+\s+){{0,3}}{_Q}(?:{_US})\s+person|qualify\s+as\s+(?:an?\s+)?{_Q}(?:{_US})\s+person|"
        r"\bitar\b[^.]{0,80}(?:requir|restrict|citizen|person|permanent\s+resident)|"
        r"(?:requir|restrict|citizen|person|permanent\s+resident)[^.]{0,80}\bitar\b|"
        r"access\s+to\s+(?:\w+\s+){0,3}(?:subject\s+to|controlled\s+(?:by|under))\s+(?:the\s+)?(?:u\.?s\.?\s+)?export\s+controls?|"
        r"(?:requires?|require|involves?)\s+(?:\w+\s+){0,2}(?:exposure|access)\s+to\s+export[- ]controlled|"
        r"(?:must|able\s+to)\s+(?:meet|satisfy|comply\s+with)\s+(?:all\s+)?(?:u\.?s\.?\s+)?export[- ]control\s+requirements|"
        r"eligible\s+to\s+(?:apply\s+for\s+and\s+)?obtain\s+(?:an?\s+|the\s+)?(?:appropriate\s+)?export\s+control\s+license", re.I),
    "no_sponsor": re.compile(
        r"(?:not|unable\s+to|cannot|can't|won't|will\s+not)\s+(?:\w+\s+){0,3}sponsor|"
        r"sponsorship\s+(?:is\s+)?not\s+(?:available|offered|provided)|no\s+(?:visa\s+)?sponsorship|"
        r"sponsorship\s+available\?\s*no\b|"
        r"without\s+(?:the\s+need\s+for\s+)?(?:current\s+or\s+future\s+)?(?:visa\s+|employer\s+|work\s+visa\s+)?sponsorship|"
        r"(?:does|do)\s+not\s+require\s+(?:\w+\s+){0,2}sponsorship", re.I),
}
MAYBE = re.compile(r"certain\s+(?:positions|roles)|some\s+(?:positions|roles)|may\s+be\s+subject", re.I)


def restrictions(text):
    """Work-authorization limits a posting states, e.g. ["citizen", "clearance"]."""
    found = []
    for kind, rx in RESTRICT.items():
        for m in rx.finditer(text or ""):
            # Look back a fixed span, not to the last period: "U.S.C. § 1158" is full of periods.
            if not MAYBE.search(text[max(0, m.start() - 350): m.end()]):
                found.append(kind)
                break
    return found


# European postings put a gender tag in the title: (m/w/d), (h/f), (f/m/x)...
GENDER_TAG = re.compile(r"\((?:[mwfdhx]\s*/\s*){1,3}[mwfdhx]\)", re.I)
# "City, Region, SA": a trailing two-letter code after two or more parts is a country, not a US state.
COUNTRY_CODE = re.compile(r",[^,]*,\s*([A-Z]{2})\s*$")
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


def load_closed_ids():
    """Jobs found closed before: the pipeline's closed_jobs.csv + our own cache."""
    ids = set()
    path = PIPELINE / "data" / "closed_jobs.csv"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):     # by id and by link (the tracker may carry another source's id)
                ids |= {r["job_id"], r.get("job_url", "").split("?")[0]}
    return ids | {k for k, (st, _) in read_cache().items() if st == "closed"}


def read_cache():
    if not CACHE.exists():
        return {}
    with open(CACHE, encoding="utf-8") as f:
        return {r["job_id"]: (r["state"], r["checked"]) for r in csv.DictReader(f)}


def check_liveness(jobs):
    """{job_id: open|closed|unknown} via job-pipeline's src/liveness.check (imported read-only).
    Each job is checked at most once a day (cache/liveness.csv); one thread per host, ~1 request/s."""
    sys.dont_write_bytecode = True                     # don't leave .pyc files in the pipeline folder
    sys.path.insert(0, str(PIPELINE))
    import requests
    from src import liveness

    cache, today = read_cache(), date.today().isoformat()
    todo = [j for j in jobs if cache.get(j["id"], ("", ""))[1] != today]
    by_host = defaultdict(list)
    for j in todo:
        by_host[urlparse(j["url"]).netloc.lower()].append(j)
    lock = threading.Lock()

    def host_worker(items):
        s = requests.Session()
        for j in items:
            try:
                st = liveness.check({"job_url": j["url"], "date_posted": j["posted"], "first_seen": j["first_seen"]}, s)
            except Exception:
                st = "unknown"
            with lock:
                cache[j["id"]] = (st, today)
            time.sleep(1)

    if todo:
        print(f"checking {len(todo)} links on {len(by_host)} hosts…", flush=True)
        with ThreadPoolExecutor(16) as pool:
            list(pool.map(host_worker, by_host.values()))
        CACHE.parent.mkdir(exist_ok=True)
        with open(CACHE, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["job_id", "state", "checked"])
            w.writerows([k, st, d] for k, (st, d) in sorted(cache.items()))
    states = {j["id"]: cache.get(j["id"], ("unknown", ""))[0] for j in jobs}
    print("liveness:", dict(Counter(states.values())))
    return states


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
    ap.add_argument("--no-check", action="store_true", help="skip the live-link check (cached 'closed' still applies)")
    args = ap.parse_args()
    csv.field_size_limit(10**9)
    vocab, sponsor = load_vocab(), sponsor_lookup(load_sponsors())
    cutoff = (date.today() - timedelta(days=args.max_age)).isoformat()
    closed_ids = load_closed_ids()
    candidates, kept, seen_keys, texts = [], [], set(), {}
    skipped = {"source": 0, "not_ra_qa": 0, "old": 0, "non_us": 0, "closed": 0, "dup": 0}

    for job_id, (r, first_seen, _last_seen) in sorted(read_raw().items(), key=lambda kv: kv[1][2], reverse=True):
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
        code = COUNTRY_CODE.search(loc)
        if (not state and NON_US.search(loc)) or GENDER_TAG.search(r["title"]) or (code and code.group(1) != "US"):
            skipped["non_us"] += 1
            continue
        c = classify(r, vocab)
        if not c:
            skipped["not_ra_qa"] += 1
            continue
        if job_id in closed_ids or (r["job_url"] or "").split("?")[0] in closed_ids:
            skipped["closed"] += 1
            continue
        company = html.unescape(r["company"] or "").replace("\xa0", " ").strip()
        family, terms = c
        ra_qa_filings, all_filings = sponsor(company)
        title = r["title"].strip()
        texts[job_id] = f"{title}. {r['description'] or ''}"
        candidates.append({
            "id": job_id,
            "first_seen": first_seen,
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
            "restrict": restrictions(f"{title}. {r['description'] or ''}"),
        })

    states = {} if args.no_check else check_liveness(candidates)
    for j in candidates:
        if states.get(j["id"]) == "closed":
            skipped["closed"] += 1
            continue
        key = (normalize_company(j["company"]), re.sub(r"\W+", " ", j["title"].lower()).strip(), j["state"])
        if key in seen_keys:
            skipped["dup"] += 1
            continue
        seen_keys.add(key)
        del j["first_seen"]
        kept.append(j)

    kept.sort(key=lambda j: j["posted"], reverse=True)
    out = {"generated": datetime.now().isoformat(timespec="minutes"), "count": len(kept), "jobs": kept}
    (SITE / "jobs.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    fams = {f: sum(j["family"] == f for j in kept) for f in ("RA", "QA", "QC", "DOC")}
    print(f"{len(kept)} jobs -> site/jobs.json {fams}; skipped {skipped}")
    embed_jobs([{"id": j["id"], "text": texts[j["id"]]} for j in kept])


def embed_jobs(items):
    """Job vectors for the site's semantic matching (tools/embed_jobs.mjs, needs Node + `npm install`).
    Descriptions only go into the gitignored cache/; the site gets vectors, never the text."""
    CACHE.parent.mkdir(exist_ok=True)
    (CACHE.parent / "job_texts.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    try:
        subprocess.run(["node", str(ROOT / "tools" / "embed_jobs.mjs")], check=True, cwd=ROOT)
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"embedding skipped ({e}); the site falls back to keyword matching")


if __name__ == "__main__":
    main()
