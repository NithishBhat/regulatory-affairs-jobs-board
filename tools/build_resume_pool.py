"""Pick RA/QA résumés out of public résumé datasets (downloaded to cache/resumes_raw/, never committed).

    C:\\Users\\nithi\\venvs\\ra-qa-jobboard\\Scripts\\python.exe tools/build_resume_pool.py

Keeps résumés that name an RA/QA role and show regulated-industry evidence (FDA, GMP, ISO 13485, pharma…),
using the site's own vocab. Writes cache/resume_pool.json: [{id, source, family, score, terms, text}].
"""
import glob
import json
import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "cache" / "resumes_raw"
VOCAB = json.loads((ROOT / "site" / "vocab.json").read_text(encoding="utf-8"))
TERMS = [(t["id"], t["cat"], re.compile(t["pattern"], re.I)) for t in VOCAB["terms"]]
GENERIC = {"iso9001", "iso17025", "labeling", "postmarket", "intlreg", "nda", "fda", "audit", "sop"}
ROLE = {
    "RA": re.compile(r"regulatory (affairs|specialist|associate|manager|director|submissions|compliance)|\bra (specialist|manager|associate)\b", re.I),
    "QA": re.compile(r"quality (assurance|systems?|engineer|specialist|manager|compliance)|\bqa (specialist|associate|manager|lead)\b|capa|validation (engineer|specialist)", re.I),
    "QC": re.compile(r"quality control|\bqc (analyst|chemist|microbiologist|technician)|analytical chemist", re.I),
    "DOC": re.compile(r"document(ation)? control|document(ation)? specialist", re.I),
}
SOFTWARE = re.compile(r"selenium|test automation|software test|manual testing|jira|test cases|sdet|appium|cypress", re.I)


def texts():
    """(source, raw_id, text) from each dataset's text column."""
    for f in sorted(glob.glob(str(RAW / "*.parquet"))):
        name = os.path.basename(f).replace(".parquet", "")
        d = pd.read_parquet(f)
        text_col = next((c for c in ("Resume_str", "Text", "text", "resume_text", "Resume", "Resume_test") if c in d), None)
        if text_col:
            col = d[text_col]
        elif "CV" in d:                         # Djinni-style profiles: title + highlights + CV
            col = d[["Position", "Highlights", "Moreinfo", "CV"]].fillna("").astype(str).agg("\n".join, axis=1)
        elif "ground_truth" in d:
            col = d["ground_truth"].map(lambda g: json.dumps(g, default=str) if g is not None else None)
        else:                                   # structured JSON résumés
            col = d.apply(lambda r: "\n".join(str(v) for v in r.values if v is not None), axis=1)
        for i, t in enumerate(col):
            if isinstance(t, str):              # pandas 3 keeps missing values as NaN even after astype(str)
                yield name, i, t


IT = re.compile(r"\b(java|hadoop|\.net|sql (developer|server)|network engineer|cloud|azure|aws|devops|solution architect|"
                r"data center|cyber ?security|front[- ]end|full[- ]stack|salesforce)\b", re.I)
CAT = {a: c for a, c, _ in TERMS}


def clean(text):
    """Unwrap prompt-formatted copies and strip scraped-HTML leftovers ("span lManagerspan")."""
    m = re.search(r"<\|im_start\|>user\s*Resume:\s*(.*?)<\|im_end\|>", text, re.S)
    t = m.group(1) if m else text
    t = re.sub(r"\bspan l\w*span\b|\bspan\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def dedupe_key(t):
    """Same résumé appears in several datasets with a different name/contact block and punctuation."""
    body = re.sub(r"[^a-z]", "", t.lower())
    return body[300:700] if len(body) > 900 else body


def main():
    pool, seen, by_source = [], set(), Counter()
    for source, i, text in texts():
        t = clean(text)
        key = dedupe_key(t)
        if len(t) < 600 or key in seen:
            continue
        seen.add(key)
        roles = {f: len(rx.findall(t)) for f, rx in ROLE.items()}
        if not any(roles.values()):              # cheap exit before the 90 vocab regexes
            continue
        headline = {f: bool(rx.search(t[:200])) for f, rx in ROLE.items()}
        terms = sorted({tid for tid, _, rx in TERMS if rx.search(t)})
        regulated = [x for x in terms if x not in GENERIC and CAT[x] in ("standard", "submission", "industry")]
        # An RA/QA person: role in the headline with some regulated evidence, or the role repeated with a lot
        # of it. Software testers and IT people out.
        ok = (any(headline.values()) and len(regulated) >= 2) or (sum(roles.values()) >= 2 and len(regulated) >= 4)
        if not ok or len(SOFTWARE.findall(t)) > 2 or len(IT.findall(t)) > 3:
            continue
        family = max(roles, key=roles.get)
        pool.append({"id": f"{source}:{i}", "source": source, "family": family, "score": len(regulated) + sum(roles.values()),
                     "terms": terms, "text": t})
        by_source[source] += 1
    pool.sort(key=lambda r: -r["score"])
    (ROOT / "cache" / "resume_pool.json").write_text(json.dumps(pool, ensure_ascii=False), encoding="utf-8")
    print(len(pool), "RA/QA résumés;", dict(by_source), "; by family", dict(Counter(r["family"] for r in pool)))


if __name__ == "__main__":
    main()
