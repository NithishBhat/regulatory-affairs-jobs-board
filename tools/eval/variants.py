"""Cheap variants of the site's signals, on the general eval set (GPU).

    C:\\Users\\nithi\\venvs\\ra-qa-jobboard\\Scripts\\python.exe -I tools/eval/variants.py

TF-IDF settings, and ways to aggregate chunk similarities for embedding models:
  max        best résumé chunk × job chunk pair (the site today)
  top3       mean over the 3 best résumé chunks of their best job chunk
  r2j        mean over résumé chunks of their best job chunk (ColBERT-style, résumé side)
  j2r        mean over job chunks of their best résumé chunk: how much of the posting the résumé covers
  j2r+max    average of the j2r and max z-scores
at chunk sizes of 180, 90 and 45 words. Each is scored alone and in the site blend
(keyword + TF-IDF + embedding, weights by 5-fold CV). Writes tools/results/variants.md.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import EVAL, RESULTS, ROOT, bootstrap_diff, blend, cv_blend3, load_labels, per_resume, summary, zscore_all  # noqa: E402

MODELS = {"gte-base": ("thenlper/gte-base", ""), "jobbert-v2": ("TechWolf/JobBERT-v2", ""),
          "arctic-l-v2": ("Snowflake/snowflake-arctic-embed-l-v2.0", "query: ")}
SIZES = (180, 90, 45)


def chunks(t, n):
    w = str(t or "").split()
    return [" ".join(w[i:i + n]) for i in range(0, len(w), n)] or [""]


def tfidf_variants(resumes, ids, docs):
    V = {
        "word 1-2, min_df 2 (site)": dict(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=2),
        "word 1-3, min_df 2": dict(ngram_range=(1, 3), sublinear_tf=True, stop_words="english", min_df=2),
        "word 1-2, min_df 1": dict(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=1),
        "word 1-2, max_df 0.5": dict(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=2, max_df=0.5),
        "word 1-2, raw tf": dict(ngram_range=(1, 2), sublinear_tf=False, stop_words="english", min_df=2),
        "word 1, min_df 2": dict(ngram_range=(1, 1), sublinear_tf=True, stop_words="english", min_df=2),
        "char_wb 3-5": dict(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2),
    }
    out = {}
    for name, kw in V.items():
        vec = TfidfVectorizer(**kw)
        X = vec.fit_transform(docs)
        Q = vec.transform([r["text"] for r in resumes])
        S = (Q @ X.T).toarray()
        out[name] = {r["id"]: dict(zip(ids, S[i].tolist())) for i, r in enumerate(resumes)}
    return out


def emb_variants(name, resumes, ids, docs):
    mid, qp = MODELS[name]
    model = SentenceTransformer(mid, device="cuda", model_kwargs={"torch_dtype": torch.float16})
    model.max_seq_length = 512
    enc = lambda xs: model.encode(xs, batch_size=32, normalize_embeddings=True, convert_to_tensor=True).float()
    out = {}
    for size in SIZES:
        jc = [chunks(d, size) for d in docs]
        J = enc([c for x in jc for c in x])
        owner = torch.tensor([i for i, x in enumerate(jc) for _ in x], device="cuda")
        counts = torch.bincount(owner, minlength=len(docs)).float()
        res = {k: {} for k in ("max", "top3", "r2j", "j2r")}
        for r in resumes:
            R = enc([qp + c for c in chunks(r["text"], size)])
            sim = J @ R.T                                            # job chunks × résumé chunks
            best_job_chunk = sim.max(1).values                        # each job chunk's best résumé chunk
            j2r = torch.zeros(len(docs), device="cuda").index_add_(0, owner, best_job_chunk) / counts
            # per job, per résumé chunk: best job chunk
            per = torch.full((len(docs), sim.shape[1]), -1.0, device="cuda").scatter_reduce(
                0, owner.unsqueeze(1).expand_as(sim), sim, "amax")
            mx = per.max(1).values
            r2j = per.mean(1)
            top3 = per.topk(min(3, per.shape[1]), dim=1).values.mean(1)   # 3 best résumé chunks' matches
            for key, v in (("max", mx), ("top3", top3), ("r2j", r2j), ("j2r", j2r)):
                res[key][r["id"]] = dict(zip(ids, v.tolist()))
        for key, v in res.items():
            out[f"{name} {key} @{size}w"] = v
        out[f"{name} j2r+max @{size}w"] = blend([zscore_all(res["j2r"]), zscore_all(res["max"])], [0.5, 0.5])
    del model
    torch.cuda.empty_cache()
    return out


def main():
    labels = load_labels("labels")
    resumes = [r for r in json.loads((EVAL / "resumes.json").read_text(encoding="utf-8")) if r["id"] in labels]
    kind = {r["id"]: r["kind"] for r in resumes}
    real = {r for r in labels if kind.get(r) == "real"}
    jobs = json.loads((ROOT / "site" / "jobs.json").read_text(encoding="utf-8"))["jobs"]
    texts = {x["id"]: x["text"] for x in json.loads((ROOT / "cache" / "job_texts.json").read_text(encoding="utf-8"))}
    ids = [j["id"] for j in jobs if j["id"] in texts]
    docs = [texts[i] for i in ids]
    kw = json.loads((EVAL / "scores" / "site_keyword.json").read_text(encoding="utf-8"))
    site = json.loads((EVAL / "scores" / "site_final.json").read_text(encoding="utf-8"))

    tf = tfidf_variants(resumes, ids, docs)
    emb = {}
    for m in MODELS:
        emb.update(emb_variants(m, resumes, ids, docs))
        print("done", m, flush=True)

    fmt = lambda t: f"{t[0]:.3f} / {t[1]:.3f} / {t[2]:.2f}"
    rows = []
    def add(section, name, sc, blended):
        pr = per_resume(sc, labels)
        rows.append((section, name, summary(pr), summary(pr, real), blended))

    base_tf = tf["word 1-2, min_df 2 (site)"]
    for name, sc in tf.items():
        h, ws = cv_blend3(kw, sc, emb["gte-base max @180w"], labels)
        add("TF-IDF", name, sc, (summary(per_resume(h, labels)), bootstrap_diff(h, site, labels)))
    for name, sc in emb.items():
        h, ws = cv_blend3(kw, base_tf, sc, labels)
        add("Embedding", name, sc, (summary(per_resume(h, labels)), bootstrap_diff(h, site, labels)))

    md = ["# Signal variants — 40 résumés, 1,860 labeled pairs", "",
          "ρ / nDCG@10 / P@10. 'Blend' = keyword + TF-IDF + embedding, weights by 5-fold CV over résumés "
          "(TF-IDF rows use gte-base max @180w; embedding rows use the site's TF-IDF). "
          "Δρ is vs the site's shown scores (keyword + TF-IDF + gte-base max @180w, fixed weights).", ""]
    for section in ("TF-IDF", "Embedding"):
        md += [f"## {section}", "", "| Variant | alone (all) | alone (real) | blend (all) | Δρ vs site (95% CI) |", "|---|---|---|---|---|"]
        for s, name, a, r, (b, d) in sorted((x for x in rows if x[0] == section), key=lambda x: -x[4][0][0]):
            md.append(f"| {name} | {fmt(a)} | {fmt(r)} | {fmt(b)} | {d[0]:+.3f} ({d[1]:+.3f} … {d[2]:+.3f}) |")
        md.append("")
    (RESULTS / "variants.md").write_text("\n".join(md), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
