"""Refresh the site's job data from the job pipeline's latest output. Runs only when you start it.

    py refresh.py                 rebuild, test, and show what changed (nothing goes live)
    py refresh.py --check-links   also re-check that job links are still open (contacts employer sites)
    py refresh.py --publish       after a good rebuild, commit the data files and push (goes live)

Steps: back up the current site data -> export.py (reads the pipeline's files; never runs or changes the
pipeline) -> sanity checks -> tests -> summary. If a check or test fails, the previous data is restored and
nothing is published. --publish commits only the data files (jobs.json, lexical.*, vectors.*), never code.
Each run appends one line to cache/refresh.log.
"""
import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
BACKUP = ROOT / "cache" / "site_backup"
LOG = ROOT / "cache" / "refresh.log"
DATA = ["jobs.json", "lexical.json", "lexical.bin", "vectors.json", "vectors.bin"]
# Backed up and restored with the site data; job_texts.json is the job text the indexes and tests are built from.
FILES = [SITE / f for f in DATA] + [ROOT / "cache" / "job_texts.json"]
MIN_JOBS, MIN_KEEP = 100, 0.6       # refuse a rebuild under 100 jobs, or one that loses 40%+ of them


def run(cmd, **kw):
    print("  $", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, cwd=ROOT, **kw)


def load_jobs():
    p = SITE / "jobs.json"
    return json.loads(p.read_text(encoding="utf-8"))["jobs"] if p.exists() else []


def backup():
    BACKUP.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        if f.exists():
            shutil.copy2(f, BACKUP / f.name)


def restore():
    for f in FILES:
        if (BACKUP / f.name).exists():
            shutil.copy2(BACKUP / f.name, f)
    print("Restored the previous site data.")


def fail(msg, entry):
    print("FAILED:", msg)
    restore()
    entry.update(ok=False, error=msg)
    log(entry)
    sys.exit(1)


def log(entry):
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def publish(summary):
    files = [f"site/{f}" for f in DATA]
    if run(["git", "add", *files]).returncode:
        sys.exit("git add failed")
    if not run(["git", "diff", "--cached", "--quiet", "--", *files]).returncode:
        print("No data changes to publish.")
        return
    msg = f"Data refresh {datetime.now():%Y-%m-%d}: {summary['jobs']} jobs (+{summary['added']} / -{summary['removed']})"
    if run(["git", "commit", "-m", msg, "--", *files]).returncode or run(["git", "push"]).returncode:
        sys.exit("Commit or push failed; the rebuilt data is still in site/.")
    print("Published:", msg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-links", action="store_true", help="re-check job links (contacts employer sites)")
    ap.add_argument("--publish", action="store_true", help="commit the data files and push (goes live)")
    ap.add_argument("--skip-tests", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    entry = {"at": datetime.now().isoformat(timespec="seconds"), "check_links": args.check_links}

    before = load_jobs()
    backup()
    print(f"1/4 Rebuilding from the pipeline's files ({len(before)} jobs now)…")
    cmd = [sys.executable, "export.py"] + ([] if args.check_links else ["--no-check"])
    if run(cmd).returncode:
        fail("export.py failed", entry)

    after = load_jobs()
    print("2/4 Sanity checks…")
    if len(after) < MIN_JOBS:
        fail(f"only {len(after)} jobs (minimum {MIN_JOBS})", entry)
    if before and len(after) < MIN_KEEP * len(before):
        fail(f"job count fell from {len(before)} to {len(after)}", entry)

    if not args.skip_tests:
        print("3/4 Tests…")
        for name, cmd in (("site", ["node", "--test", "tools/test/match.test.mjs"]),
                          ("export", [sys.executable, "tools/test/test_export.py"])):
            r = run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if r.returncode:
                lines = (r.stdout + r.stderr).splitlines()
                print("\n".join(l for l in lines if l.startswith("not ok") or "expected" in l or "actual" in l
                                or "Error" in l)[-2000:])
                fail(f"{name} tests failed", entry)

    old, new = {j["id"] for j in before}, {j["id"] for j in after}
    fams = {f: sum(j["family"] == f for j in after) for f in ("RA", "QA", "QC", "DOC")}
    summary = {"jobs": len(after), "added": len(new - old), "removed": len(old - new), "families": fams}
    entry.update(ok=True, **summary)
    print(f"4/4 Done: {len(after)} jobs (+{summary['added']} new, -{summary['removed']} gone) {fams}")

    if args.publish:
        publish(summary)
        entry["published"] = True
    else:
        print("Not published. Check it on localhost, then run: py refresh.py --publish")
    log(entry)


if __name__ == "__main__":
    main()
