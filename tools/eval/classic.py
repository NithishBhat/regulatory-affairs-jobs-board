"""Classic (pre-transformer) text vectors on the general eval set.

    C:\\Users\\nithi\\venvs\\ra-qa-jobboard\\Scripts\\python.exe -I tools/eval/classic.py

  lsa-<k>        TF-IDF (site settings) -> truncated SVD with k dims, fit on the jobs
  glove-300      pretrained GloVe (Wikipedia + Gigaword, 300d), IDF-weighted mean of word vectors
  w2v / d2v      word2vec (IDF-weighted mean) and doc2vec trained on the jobs + public résumé corpus
                 (cache/resumes_raw; unsupervised, no labels)
Cosine résumé × job. Writes scores/classic_<name>.json; blend with tools/eval/combine.py.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import EVAL, RESULTS, ROOT, load_labels, per_resume, summary  # noqa: E402

TOK = re.compile(r"[a-z0-9]{2,}")
tok = lambda t: [w for w in TOK.findall(str(t).lower()) if w not in ENGLISH_STOP_WORDS]


def corpus(limit=60000):
    texts = []
    for f in sorted((ROOT / "cache" / "resumes_raw").glob("*.parquet")):
        d = pd.read_parquet(f)
        col = next((c for c in ("Resume_str", "Text", "text", "resume_text", "Resume", "Resume_test") if c in d), None)
        if col:
            texts += [t for t in d[col].tolist() if isinstance(t, str) and len(t) > 300]
    return texts[:limit]


def cos_matrix(R, J):
    R = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)
    J = J / (np.linalg.norm(J, axis=1, keepdims=True) + 1e-9)
    return R @ J.T


def mean_vec(tokens, kv, idf):
    vs = [kv[w] * idf.get(w, 1.0) for w in tokens if w in kv]
    return np.mean(vs, axis=0) if vs else np.zeros(kv.vector_size)


def main():
    labels = load_labels("labels")
    resumes = [r for r in json.loads((EVAL / "resumes.json").read_text(encoding="utf-8")) if r["id"] in labels]
    kind = {r["id"]: r["kind"] for r in resumes}
    real = {r for r in labels if kind.get(r) == "real"}
    jobs = json.loads((ROOT / "site" / "jobs.json").read_text(encoding="utf-8"))["jobs"]
    texts = {x["id"]: x["text"] for x in json.loads((ROOT / "cache" / "job_texts.json").read_text(encoding="utf-8"))}
    ids = [j["id"] for j in jobs if j["id"] in texts]
    docs = [texts[i] for i in ids]
    rtexts = [r["text"] for r in resumes]
    S = {}

    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=2)
    X = vec.fit_transform(docs)
    Q = vec.transform(rtexts)
    for k in (100, 200, 300):
        svd = TruncatedSVD(k, random_state=0).fit(X)
        S[f"lsa-{k}"] = cos_matrix(svd.transform(Q), svd.transform(X))

    jt, rt = [tok(d) for d in docs], [tok(t) for t in rtexts]
    df = Counter(w for d in jt for w in set(d))
    idf = {w: np.log((1 + len(jt)) / (1 + c)) + 1 for w, c in df.items()}

    import gensim.downloader as api
    from gensim.models import Doc2Vec, Word2Vec
    from gensim.models.doc2vec import TaggedDocument
    glove = api.load("glove-wiki-gigaword-300")
    S["glove-300"] = cos_matrix(np.array([mean_vec(t, glove, idf) for t in rt]),
                                np.array([mean_vec(t, glove, idf) for t in jt]))
    del glove

    extra = [tok(t) for t in corpus()]
    print(len(extra), "corpus résumés", flush=True)
    train = jt + extra
    w2v = Word2Vec(train, vector_size=200, window=8, min_count=3, workers=8, epochs=10, seed=0)
    S["w2v-200"] = cos_matrix(np.array([mean_vec(t, w2v.wv, idf) for t in rt]),
                              np.array([mean_vec(t, w2v.wv, idf) for t in jt]))
    d2v = Doc2Vec([TaggedDocument(t, [i]) for i, t in enumerate(train)], vector_size=200, window=8,
                  min_count=3, workers=8, epochs=20, seed=0, dm=0, dbow_words=1)
    S["d2v-200"] = cos_matrix(np.array([d2v.infer_vector(t, epochs=50) for t in rt]),
                              np.array([d2v.dv[i] for i in range(len(jt))]))

    fmt = lambda t: f"{t[0]:.3f} / {t[1]:.3f} / {t[2]:.2f}"
    lines = ["# Classic text vectors — alone", "", "ρ / nDCG@10 / P@10.", "", "| Method | all | real |", "|---|---|---|"]
    for name, M in S.items():
        sc = {r["id"]: dict(zip(ids, M[i].tolist())) for i, r in enumerate(resumes)}
        (EVAL / "scores" / f"classic_{name}.json").write_text(json.dumps(sc))
        pr = per_resume(sc, labels)
        lines.append(f"| {name} | {fmt(summary(pr))} | {fmt(summary(pr, real))} |")
    (RESULTS / "classic.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
