"""Blend several signals with weights picked by 5-fold CV over résumés (grid, step 0.1, weights sum to 1).

    C:\\Users\\nithi\\venvs\\ra-qa-jobboard\\Scripts\\python.exe -I tools/eval/combine.py NAME=FILE[:IDX] ...

FILE is a score file in cache/general_eval/scores ({resume: {job: value}} or {resume: {job: [best, mean]}},
IDX picks the element). Each signal is z-scored per résumé over all jobs, like the site does. Prints and
appends a row to tools/results/combine.md with ρ / nDCG@10 / P@10 (all, real) and Δρ vs the site.
"""
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import EVAL, RESULTS, bootstrap_diff, load_labels, per_resume, summary, zscore_all  # noqa: E402


def load(spec):
    f, _, idx = spec.partition(":")
    d = json.loads((EVAL / "scores" / f).read_text(encoding="utf-8"))
    if idx:
        d = {r: {j: v[int(idx)] for j, v in x.items()} for r, x in d.items()}
    return zscore_all(d)


def main():
    specs = [a.split("=", 1) for a in sys.argv[1:]]
    names = [n for n, _ in specs]
    Z = [load(s) for _, s in specs]
    labels = load_labels("labels")
    labels = {r: v for r, v in labels.items() if all(r in z for z in Z)}
    kind = {r["id"]: r["kind"] for r in json.loads((EVAL / "resumes.json").read_text(encoding="utf-8"))}
    real = {r for r in labels if kind.get(r) == "real"}
    pairs = {r: [j for j in labels[r] if all(j in z[r] for z in Z)] for r in labels}
    X = {r: np.array([[z[r][j] for z in Z] for j in pairs[r]]) for r in labels}   # labeled pairs only
    lab = {r: {j: labels[r][j] for j in pairs[r]} for r in labels}
    steps = np.round(np.arange(0, 1.01, 0.1), 1)
    grid = [w for w in itertools.product(steps, repeat=len(Z) - 1) if sum(w) <= 1.0001]
    grid = [np.array(list(w) + [round(1 - sum(w), 1)]) for w in grid]
    as_scores = lambda w, rs: {r: dict(zip(pairs[r], (X[r] @ w).tolist())) for r in rs}
    rho = lambda w, rs: summary(per_resume(as_scores(w, rs), {r: lab[r] for r in rs}))[0]

    rids = sorted(labels)
    random.Random(3).shuffle(rids)
    out, ws = {}, []
    for k in range(5):
        test = [r for r in rids[k::5]]
        train = [r for r in rids if r not in test]
        best = max(grid, key=lambda w: rho(w, train))
        ws.append("/".join(f"{x:.1f}" for x in best))
        out.update(as_scores(best, test))
    site = json.loads((EVAL / "scores" / "site_final.json").read_text(encoding="utf-8"))
    pr = per_resume(out, lab)
    d = bootstrap_diff(out, site, lab)
    fmt = lambda t: f"{t[0]:.3f} / {t[1]:.3f} / {t[2]:.2f}"
    row = (f"| {' + '.join(names)} | {'; '.join(ws)} | {fmt(summary(pr))} | {fmt(summary(pr, real))} | "
           f"{d[0]:+.3f} ({d[1]:+.3f} … {d[2]:+.3f}) |")
    md = RESULTS / "combine.md"
    if not md.exists():
        md.write_text("# Signal combinations — 5-fold CV weights over résumés\n\nρ / nDCG@10 / P@10. Δρ vs the "
                      "site's shown scores.\n\n| Signals | CV weights per fold | all | real | Δρ vs site (95% CI) |\n"
                      "|---|---|---|---|---|\n", encoding="utf-8")
    with md.open("a", encoding="utf-8") as f:
        f.write(row + "\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print(row)


if __name__ == "__main__":
    main()
