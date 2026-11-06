"""Does an experience-gap penalty after the blend rank jobs better? 5-fold CV over the 40 eval résumés.

    node tools/eval/site_scores.mjs && node tools/eval/exp_gap.mjs
    C:\\Users\\nithi\\venvs\\ra-qa-jobboard\\Scripts\\python.exe -I tools/eval/exp_penalty.py

penalty = min(cap, per_year × gap) when the job asks for more years than the résumé shows (half weight when the
years come from the title level rather than the posting), plus a fixed penalty for director roles under 8 years.
Writes tools/results/exp_penalty.md.
"""
import itertools
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import EVAL, RESULTS, bootstrap_diff, load_labels, per_resume, summary  # noqa: E402

S = EVAL / "scores"


def apply(final, gaps, per_year, cap, inferred, director):
    out = {}
    for rid, d in final.items():
        g = gaps.get(rid, {})
        o = {}
        for jid, sc in d.items():
            gap, stated, level, known = g.get(jid, [0, 1, "mid", 0])
            if not known:            # résumé years unknown (no dates): no penalty
                o[jid] = sc
                continue
            pen = min(cap, per_year * max(0.0, gap)) * (1 if stated else inferred)
            if level == "director" and gap > 2:
                pen = max(pen, director)
            o[jid] = sc - pen
        out[rid] = o
    return out


def main():
    labels = load_labels("labels")
    kind = {r["id"]: r["kind"] for r in json.loads((EVAL / "resumes.json").read_text(encoding="utf-8"))}
    real = {r for r in labels if kind.get(r) == "real"}
    final = json.loads((S / "site_final.json").read_text(encoding="utf-8"))
    gaps = json.loads((S / "site_expgap.json").read_text(encoding="utf-8"))
    grid = list(itertools.product([0, 2, 4, 6, 8, 10, 12, 15], [10, 20, 30, 45], [0, 0.5, 1.0], [0, 15, 30]))
    rho = lambda sc, lab: (lambda s: s[0] + s[1])(summary(per_resume(sc, lab)))   # ρ + nDCG@10
    rids = sorted(labels)
    random.Random(3).shuffle(rids)
    cv, picks = {}, []
    for k in range(5):
        test = set(rids[k::5])
        train = {r: labels[r] for r in rids if r not in test}
        best = max(grid, key=lambda p: rho(apply({r: final[r] for r in train}, gaps, *p), train))
        picks.append(best)
        cv.update({r: v for r, v in apply({r: final[r] for r in test}, gaps, *best).items()})
    full_best = max(grid, key=lambda p: rho(apply(final, gaps, *p), labels))
    fmt = lambda t: f"{t[0]:.3f} / {t[1]:.3f} / {t[2]:.2f}"
    rows = [("Site now (no penalty)", final), ("Penalty, CV-picked per fold", cv),
            (f"Penalty {full_best} (picked on all 40; optimistic)", apply(final, gaps, *full_best))]
    md = ["# Experience-gap penalty after the blend — 40 résumés", "",
          "ρ / nDCG@10 / P@10. Params = (points per missing year, cap, weight when years come from the title, "
          "director under-8-years penalty). Only résumés whose years are known (dates or a stated number) are penalised.", "", "| Variant | all | real | Δρ vs site (95% CI) |", "|---|---|---|---|"]
    for name, sc in rows:
        pr = per_resume(sc, labels)
        d = bootstrap_diff(sc, final, labels)
        md.append(f"| {name} | {fmt(summary(pr))} | {fmt(summary(pr, real))} | {d[0]:+.3f} ({d[1]:+.3f} … {d[2]:+.3f}) |")
    md += ["", f"CV picks per fold: {picks}"]
    (RESULTS / "exp_penalty.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
