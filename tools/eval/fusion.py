"""Score-fusion methods (TalentCLEF 2026 top teams: per-query min-max, equal weights, subset search).

    C:\\Users\\nithi\\venvs\\ra-qa-jobboard\\Scripts\\python.exe -I tools/eval/fusion.py

Components: keyword, TF-IDF (site files) and best-chunk embeddings (scores/emb_*.json). Methods:
  z-weighted   the site: z-scores, weights by CV
  minmax-sum   per-résumé min-max, equal-weight sum
  minmax-geo   geometric mean of min-max scores
  rrf          reciprocal rank fusion, k=60
  z-equal      equal-weight z-scores
Subset search: on each training fold pick the equal-weight subset (size 2–5) with the best ρ, apply it to the
held-out résumés. Writes tools/results/fusion.md.
"""
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import EVAL, RESULTS, bootstrap_diff, load_labels, per_resume, summary, zscore_all  # noqa: E402


def minmax(scores):
    out = {}
    for rid, d in scores.items():
        v = np.array(list(d.values()), float)
        lo, hi = v.min(), v.max()
        out[rid] = {j: (x - lo) / ((hi - lo) or 1) for j, x in d.items()}
    return out


def ranks(scores):
    out = {}
    for rid, d in scores.items():
        order = sorted(d, key=d.get, reverse=True)
        out[rid] = {j: i + 1 for i, j in enumerate(order)}
    return out


def fuse(parts, method):
    rids = set.intersection(*(set(p) for p in parts))
    if method == "rrf":
        rs = [ranks(p) for p in parts]
        return {r: {j: sum(1 / (60 + x[r][j]) for x in rs) for j in parts[0][r]} for r in rids}
    if method == "z-equal":
        zs = [zscore_all(p) for p in parts]
        return {r: {j: sum(z[r][j] for z in zs) for j in parts[0][r]} for r in rids}
    ms = [minmax(p) for p in parts]
    if method == "minmax-sum":
        return {r: {j: sum(m[r][j] for m in ms) for j in parts[0][r]} for r in rids}
    if method == "minmax-geo":
        return {r: {j: float(np.prod([m[r][j] + 1e-3 for m in ms])) for j in parts[0][r]} for r in rids}
    raise ValueError(method)


def main():
    labels = load_labels("labels")
    kind = {r["id"]: r["kind"] for r in json.loads((EVAL / "resumes.json").read_text(encoding="utf-8"))}
    real = {r for r in labels if kind.get(r) == "real"}
    load = lambda n: json.loads((EVAL / "scores" / n).read_text(encoding="utf-8"))
    C = {"keyword": load("site_keyword.json"), "tfidf": load("site_lexical.json")}
    for f in sorted((EVAL / "scores").glob("emb_*.json")):
        C[f.stem[4:]] = {r: {j: v[0] for j, v in x.items()} for r, x in json.loads(f.read_text(encoding="utf-8")).items()}
    site = load("site_final.json")
    labels = {r: v for r, v in labels.items() if all(r in c for c in C.values())}
    rho = lambda sc, lab=labels: summary(per_resume(sc, lab))[0]

    fmt = lambda t: f"{t[0]:.3f} / {t[1]:.3f} / {t[2]:.2f}"
    rows = []
    def add(name, sc):
        pr = per_resume(sc, labels)
        d = bootstrap_diff(sc, site, labels)
        rows.append((name, summary(pr), summary(pr, real), d))

    add("Site (z-weighted 0.3/0.3/0.4, gte-base in browser)", site)
    for emb in ("gte-base", "jobbert-v2", "arctic-l-v2", "qwen3-emb-0.6b"):
        for m in ("minmax-sum", "minmax-geo", "rrf", "z-equal"):
            add(f"keyword + TF-IDF + {emb}: {m}", fuse([C["keyword"], C["tfidf"], C[emb]], m))

    # Subset search with equal-weight fusion, CV over résumés.
    pool = ["keyword", "tfidf"] + sorted((k for k in C if k not in ("keyword", "tfidf")),
                                         key=lambda k: -rho(C[k]))[:8]
    for method in ("minmax-sum", "z-equal"):
        cache = {}
        def fused(sub):
            if sub not in cache:
                cache[sub] = fuse([C[k] for k in sub], method)
            return cache[sub]
        subsets = [s for n in range(2, 6) for s in itertools.combinations(pool, n)]
        rids = sorted(labels)
        random.Random(3).shuffle(rids)
        out, picked = {}, []
        for k in range(5):
            test = set(rids[k::5])
            train = {r: labels[r] for r in rids if r not in test}
            best = max(subsets, key=lambda s: rho(fused(s), train))
            picked.append("+".join(best))
            out.update({r: v for r, v in fused(best).items() if r in test})
        add(f"Subset search, {method} (CV picks: {'; '.join(picked)})", out)

    rows.sort(key=lambda x: -x[1][0])
    md = ["# Fusion methods — 40 résumés", "", "ρ / nDCG@10 / P@10; Δρ vs the site's shown scores.", "",
          "| Method | all | real | Δρ vs site (95% CI) |", "|---|---|---|---|"]
    md += [f"| {n} | {fmt(a)} | {fmt(r)} | {d[0]:+.3f} ({d[1]:+.3f} … {d[2]:+.3f}) |" for n, a, r, d in rows]
    (RESULTS / "fusion.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
