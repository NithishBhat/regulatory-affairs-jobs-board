"""Fit the score calibration: blended score (s.rank) -> expected rubric score (0–100), monotone.

    node tools/eval/site_scores.mjs --reuse-semantic
    C:\\Users\\nithi\\venvs\\ra-qa-jobboard\\Scripts\\python.exe -I tools/eval/calibrate.py

Isotonic regression on the 1,860 labeled pairs, checked with 5-fold CV over résumés (mean absolute error and
tier agreement against the labels), then refit on all pairs. Two tables, because the blend differs before and
after the AI model loads: "full" (keyword + TF-IDF + embeddings) and "lexical" (keyword + TF-IDF). Written to
site/calibration.json as {full: {x, y}, lexical: {x, y}}; match.js interpolates. Writes tools/results/calibration.md.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import EVAL, RESULTS, ROOT, load_labels  # noqa: E402

TIER = lambda v: 3 if v >= 85 else 2 if v >= 70 else 1 if v >= 50 else 0      # rubric tiers
OLD_TIER = lambda v: 2 if v >= 68 else 1 if v >= 50 else 0                     # site's old cut-offs


def pairs(rank, labels, rids):
    x, y = [], []
    for r in rids:
        for j, l in labels[r].items():
            if j in rank.get(r, {}):
                x.append(rank[r][j]); y.append(l)
    return np.array(x), np.array(y, float)


def iso():
    return IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0, y_max=100)


def knots(model, x):
    qs = np.unique(np.quantile(x, np.linspace(0, 1, 25)))
    ys = np.maximum.accumulate(model.predict(qs))
    return {"x": [round(float(v), 2) for v in qs], "y": [round(float(v), 1) for v in ys]}


def fit(labels, rank):
    rids = sorted(r for r in labels if r in rank)
    random.Random(3).shuffle(rids)
    err_raw, err_cal, tier_raw, tier_cal, top = [], [], [], [], []
    for k in range(5):
        test = rids[k::5]
        train = [r for r in rids if r not in test]
        model = iso().fit(*pairs(rank, labels, train))
        x, y = pairs(rank, labels, test)
        cal, clipped = model.predict(x), np.clip(x, 0, 100)
        err_raw += list(np.abs(clipped - y)); err_cal += list(np.abs(cal - y))
        tier_raw += [OLD_TIER(a) == min(TIER(b), 2) for a, b in zip(clipped, y)]
        tier_cal += [TIER(a) == TIER(b) for a, b in zip(cal, y)]
        top += [b for a, b in zip(cal, y) if a >= 70]
    x, y = pairs(rank, labels, rids)
    table = knots(iso().fit(x, y), x)
    md = ["| Shown score | mean abs. error vs label | tier agrees with label |", "|---|---|---|",
          f"| Before (blended, capped at 100; old cut-offs 68 / 50) | {np.mean(err_raw):.1f} | {np.mean(tier_raw):.0%} |",
          f"| Calibrated (rubric tiers 85 / 70 / 50) | {np.mean(err_cal):.1f} | {np.mean(tier_cal):.0%} |", "",
          f"Held-out pairs shown at 70+: {len(top)}; label 70+ for {np.mean([b >= 70 for b in top]):.0%} "
          f"(mean label {np.mean(top):.0f}).", "",
          "Table (x = blended score, y = shown score):", "",
          "| x | " + " | ".join(f"{v:.0f}" for v in table["x"]) + " |",
          "|---|" + "---|" * len(table["x"]),
          "| y | " + " | ".join(f"{v:.0f}" for v in table["y"]) + " |"]
    return table, md


def main():
    labels = load_labels("labels")
    load = lambda f: json.loads((EVAL / "scores" / f).read_text(encoding="utf-8"))
    out = {}
    md = ["# Score calibration — blended score -> expected rubric score", "",
          "5-fold CV over résumés on the labeled pairs (labels: tools/eval/rubric.md, 85+ strong, 70–84 good, "
          "50–69 partial)."]
    for key, f, title in (("full", "site_final.json", "After the AI model loads (keyword + TF-IDF + embeddings)"),
                          ("lexical", "site_final_no_semantic.json", "Before it loads (keyword + TF-IDF)")):
        out[key], part = fit(labels, load(f))
        md += ["", f"## {title}", ""] + part
    (ROOT / "site" / "calibration.json").write_text(json.dumps(out), encoding="utf-8")
    (RESULTS / "calibration.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
