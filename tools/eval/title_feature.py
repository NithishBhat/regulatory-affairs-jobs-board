"""Title-match feature: JobBERT (trained on job titles) similarity between the posting title and the best short
window of the résumé (8 words, stride 4), so past job titles anywhere in the résumé can match.

    C:\\Users\\nithi\\venvs\\ra-qa-jobboard\\Scripts\\python.exe -I tools/eval/title_feature.py

Scored alone and added to the site blend (keyword 0.3 / TF-IDF 0.3 / gte-base 0.4) with its weight picked by
5-fold CV over résumés. Writes scores/title_<model>.json and tools/results/title_feature.md.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import EVAL, RESULTS, ROOT, blend, bootstrap_diff, load_labels, per_resume, summary, zscore_all  # noqa: E402

MODELS = {"jobbert-v2": "TechWolf/JobBERT-v2", "jobbert-v3": "TechWolf/JobBERT-v3"}


def windows(t, n=8, step=4):
    w = str(t or "").split()
    return [" ".join(w[i:i + n]) for i in range(0, max(1, len(w) - n + 1), step)] or [""]


def main():
    labels = load_labels("labels")
    resumes = [r for r in json.loads((EVAL / "resumes.json").read_text(encoding="utf-8")) if r["id"] in labels]
    kind = {r["id"]: r["kind"] for r in resumes}
    real = {r for r in labels if kind.get(r) == "real"}
    jobs = json.loads((ROOT / "site" / "jobs.json").read_text(encoding="utf-8"))["jobs"]
    load = lambda n: json.loads((EVAL / "scores" / n).read_text(encoding="utf-8"))
    kw, lex, sem, site = (load(f"site_{n}.json") for n in ("keyword", "lexical", "semantic", "final"))
    zk, zl, zs = zscore_all(kw), zscore_all(lex), zscore_all(sem)
    fmt = lambda t: f"{t[0]:.3f} / {t[1]:.3f} / {t[2]:.2f}"
    lines = ["# Title-match feature — 40 résumés", "", "ρ / nDCG@10 / P@10. Δρ vs the site's shown scores.", "",
             "| Feature | alone (all) | alone (real) | site + title (all) | site + title (real) | Δρ vs site (95% CI) |",
             "|---|---|---|---|---|---|"]
    for name, mid in MODELS.items():
        model = SentenceTransformer(mid, device="cuda", model_kwargs={"torch_dtype": torch.float16})
        enc = lambda xs: model.encode(xs, batch_size=64, normalize_embeddings=True, convert_to_tensor=True).float()
        T = enc([j["title"] for j in jobs])
        title = {}
        for r in resumes:
            W = enc(windows(r["text"]))
            title[r["id"]] = dict(zip([j["id"] for j in jobs], (T @ W.T).max(1).values.tolist()))
        (EVAL / "scores" / f"title_{name}.json").write_text(json.dumps(title))
        zt = zscore_all(title)
        rids = sorted(labels)
        random.Random(3).shuffle(rids)
        out, ws = {}, []
        for k in range(5):
            test = set(rids[k::5])
            train = {r: labels[r] for r in rids if r not in test}
            best = max(np.linspace(0, 0.5, 11), key=lambda w: summary(per_resume(
                blend([zk, zl, zs, zt], [0.3, 0.3, 0.4, w]), train))[0])
            ws.append(round(float(best), 2))
            out.update({r: v for r, v in blend([zk, zl, zs, zt], [0.3, 0.3, 0.4, best]).items() if r in test})
        a, b = per_resume(title, labels), per_resume(out, labels)
        d = bootstrap_diff(out, site, labels)
        lines.append(f"| {name} title (CV w={sorted(set(ws))}) | {fmt(summary(a))} | {fmt(summary(a, real))} | "
                     f"{fmt(summary(b))} | {fmt(summary(b, real))} | {d[0]:+.3f} ({d[1]:+.3f} … {d[2]:+.3f}) |")
        del model
        torch.cuda.empty_cache()
    (RESULTS / "title_feature.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
