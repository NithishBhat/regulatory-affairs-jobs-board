"""Evaluate ranking strategies on the general labeled set.

    C:\\Users\\nithi\\venvs\\ra-qa-jobboard\\Scripts\\python.exe tools/eval/evaluate.py

Labels: cache/general_eval/labels/*.json (LLM raters, tools/eval/rubric.md); labels_b/ = second raters.
Scores: keyword_scores.json, scores/emb_*.json ([best-chunk, mean]), scores/cross_*.json.
Metrics are computed per résumé over its labeled pool and averaged (macro): Spearman ρ, nDCG@10 (gains 3/2/1/0
for ≥85 / 70–84 / 50–69 / <50) and P@10 (share of the top 10 labeled ≥70). Hybrid weights are picked by 5-fold
cross-validation over résumés; 95% CIs bootstrap over résumés. Output: tools/results/general_eval.{md,json}.
"""
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "cache" / "general_eval"
RESULTS = ROOT / "tools" / "results"
GAIN = lambda l: 3 if l >= 85 else 2 if l >= 70 else 1 if l >= 50 else 0


def load_labels(folder):
    out = {}
    for f in sorted((EVAL / folder).glob("*.json")):
        rid = json.loads((EVAL / "tasks" / f.name).read_text(encoding="utf-8"))["resume_id"]
        out[rid] = {x["job_id"]: int(x["score"]) for x in json.loads(f.read_text(encoding="utf-8"))}
    return out


def ndcg(pred, lab, k=10):
    order = np.argsort(-pred)[:k]
    ideal = np.sort([GAIN(l) for l in lab])[::-1][:k]
    dcg = sum((2 ** GAIN(lab[i]) - 1) / math.log2(r + 2) for r, i in enumerate(order))
    idcg = sum((2 ** g - 1) / math.log2(r + 2) for r, g in enumerate(ideal))
    return dcg / idcg if idcg else None


def per_resume(scores, labels):
    """{rid: (rho, ndcg10, p10)} for one strategy; scores[rid][jid] -> float."""
    out = {}
    for rid, lab in labels.items():
        jids = [j for j in lab if j in scores.get(rid, {})]
        if len(jids) < 10:
            continue
        p = np.array([scores[rid][j] for j in jids], float)
        l = np.array([lab[j] for j in jids], float)
        rho = spearmanr(p, l).statistic if np.std(p) > 0 and np.std(l) > 0 else 0.0
        top = np.argsort(-p)[:10]
        out[rid] = (float(rho), ndcg(p, l), float(np.mean(l[top] >= 70)))
    return out


def summary(pr, rids=None):
    vals = [v for r, v in pr.items() if rids is None or r in rids]
    rho = np.mean([v[0] for v in vals])
    nd = np.mean([v[1] for v in vals if v[1] is not None])
    p10 = np.mean([v[2] for v in vals])
    return float(rho), float(nd), float(p10)


def zscore_all(scores):
    """z-normalise each résumé's scores over all jobs it has (what a live system would do)."""
    out = {}
    for rid, d in scores.items():
        v = np.array(list(d.values()), float)
        m, s = v.mean(), v.std() or 1.0
        out[rid] = {j: (x - m) / s for j, x in d.items()}
    return out


def blend(parts, weights):
    rids = set.intersection(*(set(p) for p in parts))
    out = {}
    for rid in rids:
        jids = set.intersection(*(set(p[rid]) for p in parts))
        out[rid] = {j: sum(w * p[rid][j] for w, p in zip(weights, parts)) for j in jids}
    return out


def cv_blend(a, b, labels, folds=5, seed=3):
    """w·a + (1−w)·b with w picked on training résumés, applied to held-out ones."""
    za, zb = zscore_all(a), zscore_all(b)
    rids = sorted(set(labels) & set(za) & set(zb))
    random.Random(seed).shuffle(rids)
    out, ws = {}, []
    for k in range(folds):
        test = set(rids[k::folds])
        train = {r: labels[r] for r in rids if r not in test}
        best = max((summary(per_resume(blend([za, zb], [w, 1 - w]), train))[0], w) for w in np.linspace(0, 1, 11))
        ws.append(round(float(best[1]), 1))
        out.update({r: v for r, v in blend([za, zb], [best[1], 1 - best[1]]).items() if r in test})
    return out, ws


def cv_blend3(a, b, c, labels, folds=5, seed=3):
    """wa·a + wb·b + (1−wa−wb)·c on z-scores, weights picked on training résumés."""
    za, zb, zc = zscore_all(a), zscore_all(b), zscore_all(c)
    rids = sorted(set(labels) & set(za) & set(zb) & set(zc))
    random.Random(seed).shuffle(rids)
    grid = [(x, y) for x in np.linspace(0, 1, 11) for y in np.linspace(0, 1, 11) if x + y <= 1.0001]
    out, ws = {}, []
    for k in range(folds):
        test = set(rids[k::folds])
        train = {r: labels[r] for r in rids if r not in test}
        best = max((summary(per_resume(blend([za, zb, zc], [x, y, 1 - x - y]), train))[0], x, y) for x, y in grid)
        ws.append((round(float(best[1]), 1), round(float(best[2]), 1)))
        out.update({r: v for r, v in blend([za, zb, zc], [best[1], best[2], 1 - best[1] - best[2]]).items() if r in test})
    return out, ws


def learned_ranker(features, labels, folds=5, seed=3):
    """LightGBM LambdaRank on per-résumé z-scored features, grouped 5-fold CV over résumés (each résumé's
    predictions come from a model that never saw that résumé)."""
    import lightgbm as lgb
    names = list(features)
    zf = {n: zscore_all(features[n]) for n in names}
    rids = sorted(r for r in labels if all(r in zf[n] for n in names))
    random.Random(seed).shuffle(rids)

    def rows(rs):
        X, y, g, keys = [], [], [], []
        for r in rs:
            jids = [j for j in labels[r] if all(j in zf[n][r] for n in names)]
            X += [[zf[n][r][j] for n in names] for j in jids]
            y += [GAIN(labels[r][j]) for j in jids]
            g.append(len(jids))
            keys += [(r, j) for j in jids]
        return np.array(X), np.array(y), g, keys

    out, imp = {}, np.zeros(len(names))
    for k in range(folds):
        test = rids[k::folds]
        Xtr, ytr, gtr, _ = rows([r for r in rids if r not in set(test)])
        Xte, _, _, kte = rows(test)
        m = lgb.LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.03, num_leaves=15,
                           min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
        m.fit(Xtr, ytr, group=gtr)
        imp += m.booster_.feature_importance("gain")
        for (r, j), p in zip(kte, m.predict(Xte)):
            out.setdefault(r, {})[j] = float(p)
    top = sorted(zip(names, imp / imp.sum()), key=lambda t: -t[1])[:6]
    return out, top


def lexical(resumes, jobs, texts):
    """TF-IDF (uni+bigram) cosine and BM25 for every résumé × job."""
    ids = [j["id"] for j in jobs]
    docs = [texts[i] for i in ids]
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=2)
    X = vec.fit_transform(docs)
    tf = {}
    for r in resumes:
        q = vec.transform([r["text"]])
        sims = (X @ q.T).toarray().ravel()
        tf[r["id"]] = dict(zip(ids, sims.tolist()))
    tok = lambda t: re.findall(r"[a-z0-9]+", t.lower())
    dt = [tok(d) for d in docs]
    avg = np.mean([len(d) for d in dt])
    df = defaultdict(int)
    for d in dt:
        for w in set(d):
            df[w] += 1
    n = len(dt)
    tfs = [defaultdict(int) for _ in dt]
    for c, d in zip(tfs, dt):
        for w in d:
            c[w] += 1
    bm = {}
    for r in resumes:
        q = set(tok(r["text"]))
        row = {}
        for i, (c, d) in enumerate(zip(tfs, dt)):
            s = 0.0
            for w in q:
                f = c.get(w)
                if f:
                    idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                    s += idf * f * 2.5 / (f + 1.5 * (0.25 + 0.75 * len(d) / avg))
            row[ids[i]] = s
        bm[r["id"]] = row
    return tf, bm


def bootstrap_diff(a, b, labels, n=2000, seed=5):
    pa, pb = per_resume(a, labels), per_resume(b, labels)
    rids = sorted(set(pa) & set(pb))
    rng = random.Random(seed)
    d = []
    for _ in range(n):
        s = [rng.choice(rids) for _ in rids]
        d.append(np.mean([pa[r][0] - pb[r][0] for r in s]))
    d.sort()
    return float(np.mean([pa[r][0] - pb[r][0] for r in rids])), d[int(0.025 * n)], d[int(0.975 * n)]


def main():
    labels, labels_b = load_labels("labels"), load_labels("labels_b")
    resumes = json.loads((EVAL / "resumes.json").read_text(encoding="utf-8"))
    kind = {r["id"]: r["kind"] for r in resumes}
    real = {r for r in labels if kind.get(r) == "real"}
    syn = {r for r in labels if kind.get(r) == "synthetic"}
    jobs = json.loads((ROOT / "site" / "jobs.json").read_text(encoding="utf-8"))["jobs"]
    texts = {x["id"]: x["text"] for x in json.loads((ROOT / "cache" / "job_texts.json").read_text(encoding="utf-8"))}
    jobs = [j for j in jobs if j["id"] in texts]

    S = {"Keyword (site layer 1)": json.loads((EVAL / "keyword_scores.json").read_text(encoding="utf-8"))}
    S["TF-IDF uni+bigram"], S["BM25"] = lexical([r for r in resumes if r["id"] in labels], jobs, texts)
    emb = {}
    for f in sorted((EVAL / "scores").glob("emb_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        name = f.stem[4:]
        emb[name] = {r: {j: v[0] for j, v in x.items()} for r, x in d.items()}
        S[f"{name}: best chunk"] = emb[name]
        S[f"{name}: mean"] = {r: {j: v[1] for j, v in x.items()} for r, x in d.items()}
    cross = {}
    for f in sorted((EVAL / "scores").glob("cross_*.json")):
        cross[f.stem[6:]] = json.loads(f.read_text(encoding="utf-8"))
        S[f"{f.stem[6:]} (cross-encoder)"] = cross[f.stem[6:]]

    rows = []
    def add(name, scores, note=""):
        pr = per_resume(scores, labels)
        if not pr:
            return
        rows.append({"name": name, "note": note, "n": len(pr), "all": summary(pr),
                     "real": summary(pr, real) if pr.keys() & real else None,
                     "synthetic": summary(pr, syn) if pr.keys() & syn else None})

    for name, sc in S.items():
        add(name, sc)
    site = blend([zscore_all(emb["gte-small"]), zscore_all(S["Keyword (site layer 1)"])], [0.5, 0.5]) if "gte-small" in emb else None
    if site:
        add("Keyword + gte-small best chunk, 50/50 (current site)", site)
    hybrids = {}
    for name in emb:
        h, ws = cv_blend(emb[name], S["Keyword (site layer 1)"], labels)
        hybrids[name] = h
        add(f"Keyword + {name} best chunk (CV w={sorted(set(ws))})", h)
    for name, c in cross.items():
        h, ws = cv_blend(c, S["Keyword (site layer 1)"], labels)
        add(f"Keyword + {name} cross-encoder (CV w={sorted(set(ws))})", h)
    # Three-way: keyword + TF-IDF + the five best embeddings by single-model ρ.
    ranked_emb = sorted(emb, key=lambda n: -summary(per_resume(emb[n], labels))[0])[:5]
    for name in ranked_emb:
        h, ws = cv_blend3(S["Keyword (site layer 1)"], S["TF-IDF uni+bigram"], emb[name], labels)
        hybrids[f"3way:{name}"] = h
        add(f"Keyword + TF-IDF + {name} (CV w={sorted(set(ws))})", h)
    # Learned ranker over every signal (research's distill-into-a-small-ranker pattern).
    feats = {"keyword": S["Keyword (site layer 1)"], "tfidf": S["TF-IDF uni+bigram"], "bm25": S["BM25"]}
    for name in emb:
        feats[f"{name}.max"] = emb[name]
        feats[f"{name}.mean"] = S[f"{name}: mean"]
    lr, top = learned_ranker(feats, labels)
    hybrids["learned:all"] = lr
    add("Learned ranker (LightGBM LambdaRank, all signals)", lr, "top features: " + ", ".join(f"{n} {w:.0%}" for n, w in top))
    small = {k: v for k, v in feats.items() if k.split(".")[0] in ("keyword", "tfidf", "bm25", "gte-small", "gte-base", "bge-base")}
    lr2, top2 = learned_ranker(small, labels)
    hybrids["learned:small"] = lr2
    add("Learned ranker (keyword, TF-IDF, BM25, ≤base-size embeddings)", lr2, "top: " + ", ".join(f"{n} {w:.0%}" for n, w in top2))
    if cross:
        lr3, top3 = learned_ranker({**feats, **{f"cross.{n}": c for n, c in cross.items()}}, labels)
        hybrids["learned:all+cross"] = lr3
        add("Learned ranker (all signals + cross-encoders)", lr3, "top: " + ", ".join(f"{n} {w:.0%}" for n, w in top3))

    rows.sort(key=lambda r: -r["all"][0])
    # Agreement between the two LLM raters on the double-labeled résumés.
    agree = []
    for rid, lb in labels_b.items():
        la = labels.get(rid, {})
        common = [j for j in lb if j in la]
        if len(common) > 5:
            a, b = [la[j] for j in common], [lb[j] for j in common]
            agree.append((rid, spearmanr(a, b).statistic, float(np.mean(np.abs(np.array(a) - np.array(b)))),
                          float(np.mean([GAIN(x) == GAIN(y) for x, y in zip(a, b)]))))
    # Bootstrap the top few against the current site.
    cis = []
    if site:
        by_label = {}
        for n, h in hybrids.items():
            for r in rows:
                if (n.startswith("learned:") and r["name"].startswith("Learned ranker")) or \
                   (n.startswith("3way:") and r["name"].startswith(f"Keyword + TF-IDF + {n[5:]} ")) or \
                   r["name"].startswith(f"Keyword + {n} best chunk"):
                    by_label.setdefault(r["name"], h)
        for r in rows[:6]:
            if r["name"] in by_label:
                cis.append((r["name"], *bootstrap_diff(by_label[r["name"]], site, labels)))
    ci = cis[0] if cis else None

    fmt = lambda t: "—" if t is None else f"{t[0]:.3f} / {t[1]:.3f} / {t[2]:.2f}"
    md = [f"# General eval — {len(labels)} résumés ({len(real)} real, {len(syn)} synthetic), "
          f"{sum(len(v) for v in labels.values())} labeled pairs", "",
          "Per-résumé ρ / nDCG@10 / P@10, averaged. Labels: LLM raters with tools/eval/rubric.md over pooled jobs.", ""]
    if agree:
        md += [f"Rater agreement on {len(agree)} double-labeled résumés: mean ρ {np.mean([a[1] for a in agree]):.3f}, "
               f"mean |Δscore| {np.mean([a[2] for a in agree]):.1f}, same tier {np.mean([a[3] for a in agree]):.0%}.", ""]
    if cis:
        md += ["Top strategies vs current site (Δρ, 95% bootstrap CI over résumés):", ""]
        md += [f"- {c[0]}: {c[1]:+.3f} ({c[2]:+.3f} … {c[3]:+.3f})" for c in cis] + [""]
    md += ["| # | Strategy | all | real | synthetic | note |", "|---|---|---|---|---|---|"]
    md += [f"| {i + 1} | {r['name']} | {fmt(r['all'])} | {fmt(r['real'])} | {fmt(r['synthetic'])} | {r['note']} |" for i, r in enumerate(rows)]
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "general_eval.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (RESULTS / "general_eval.json").write_text(json.dumps({"rows": rows, "agreement": agree, "cis": cis}, indent=1), encoding="utf-8")
    print("\n".join(md[:16 + min(len(rows), 25)]))


if __name__ == "__main__":
    main()
