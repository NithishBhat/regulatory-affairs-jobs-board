"""Score résumé × job pairs with Hugging Face models on the GPU (general eval set).

    C:\\Users\\nithi\\venvs\\ra-qa-jobboard\\Scripts\\python.exe tools/eval/gpu_models.py [--cross] [--only name,...]

Embeddings: every résumé × every job (cheap) -> cache/general_eval/scores/emb_<name>.json
            {resume_id: {job_id: [best_chunk_cosine, mean_cosine]}}
--cross:    cross-encoders on the labeled pairs only (cache/general_eval/labels.json)
            -> scores/cross_<name>.json {resume_id: {job_id: best_chunk_logit}}
Existing score files are kept. Chunking matches site/match.js (180 words).
"""
import argparse
import json
import time
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "cache" / "general_eval"
OUT = EVAL / "scores"
CHUNK = 180
BGE_Q = "Represent this sentence for searching relevant passages: "
QWEN_Q = "Instruct: Given a candidate's résumé, retrieve job postings the candidate is qualified for\nQuery: "

# name: (model id, query prefix, passage prefix, extra kwargs). General models, then job/résumé-specific ones.
# Models that need trust_remote_code (nomic-embed, gte-Qwen2) are left out: they run code from the model repo.
EMB = {
    "MiniLM-L6": ("sentence-transformers/all-MiniLM-L6-v2", "", "", {}),
    "mpnet-base": ("sentence-transformers/all-mpnet-base-v2", "", "", {}),
    "gte-small": ("thenlper/gte-small", "", "", {}),
    "gte-base": ("thenlper/gte-base", "", "", {}),
    "gte-large": ("thenlper/gte-large", "", "", {}),
    "bge-small": ("BAAI/bge-small-en-v1.5", BGE_Q, "", {}),
    "bge-base": ("BAAI/bge-base-en-v1.5", BGE_Q, "", {}),
    "bge-large": ("BAAI/bge-large-en-v1.5", BGE_Q, "", {}),
    "bge-m3": ("BAAI/bge-m3", "", "", {}),
    "e5-large": ("intfloat/e5-large-v2", "query: ", "passage: ", {}),
    "me5-large-instruct": ("intfloat/multilingual-e5-large-instruct", QWEN_Q, "", {}),
    "mxbai-large": ("mixedbread-ai/mxbai-embed-large-v1", BGE_Q, "", {}),
    "arctic-l-v2": ("Snowflake/snowflake-arctic-embed-l-v2.0", "query: ", "", {}),
    "qwen3-emb-0.6b": ("Qwen/Qwen3-Embedding-0.6B", QWEN_Q, "", {}),
    "jobbert-v2": ("TechWolf/JobBERT-v2", "", "", {}),
    "jobbert-v3": ("TechWolf/JobBERT-v3", "", "", {}),
    "talentclef-biencoder": ("talentguide/talentclef-biencoder-v1", "", "", {}),
    "resume-matcher-minilm": ("anass1209/resume-job-matcher-all-MiniLM-L6-v2", "", "", {}),
    "resume-matching-sbert": ("zoraizbinsamee/resume-job-matching-sbert", "", "", {}),
    "resume-jd-mpnet": ("dlepighe1/resume-jd-matcher-mpnet", "", "", {}),
    "resume-screener-bge": ("Amirktk0312/resume-screener-bge-finetuned", "", "", {}),
}
CROSS = {
    "msmarco-MiniLM": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge-reranker-base": "BAAI/bge-reranker-base",
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
    "qwen3-reranker-0.6b": "tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
    "resume-jd-cross": "alimuhammad24/resume-jd-cross-encoder",
}


def chunks(t):
    w = str(t or "").split()
    return [" ".join(w[i:i + CHUNK]) for i in range(0, len(w), CHUNK)] or [""]


def load():
    resumes = []
    for f in ("resumes_real.json", "resumes_synthetic.json"):
        if (EVAL / f).exists():
            resumes += json.loads((EVAL / f).read_text(encoding="utf-8"))
    jobs = json.loads((ROOT / "site" / "jobs.json").read_text(encoding="utf-8"))["jobs"]
    texts = {x["id"]: x["text"] for x in json.loads((ROOT / "cache" / "job_texts.json").read_text(encoding="utf-8"))}
    return resumes, [j for j in jobs if j["id"] in texts], texts


def embeddings(name, resumes, jobs, texts):
    mid, qp, pp, kw = EMB[name]
    t0 = time.time()
    model = SentenceTransformer(mid, device="cuda", model_kwargs={"torch_dtype": torch.float16}, **kw)
    model.max_seq_length = min(512, model.max_seq_length or 512)
    enc = lambda xs: model.encode(xs, batch_size=16, normalize_embeddings=True, convert_to_tensor=True).float()
    job_chunks = [chunks(texts[j["id"]]) for j in jobs]
    flat = enc([pp + c for jc in job_chunks for c in jc])
    owner = torch.tensor([i for i, jc in enumerate(job_chunks) for _ in jc], device="cuda")
    counts = torch.bincount(owner, minlength=len(jobs)).unsqueeze(1).float()
    job_mean = torch.nn.functional.normalize(torch.zeros(len(jobs), flat.shape[1], device="cuda").index_add_(0, owner, flat) / counts, dim=1)
    out = {}
    for r in resumes:
        rv = enc([qp + c for c in chunks(r["text"])])
        best = torch.full((len(jobs),), -1.0, device="cuda").scatter_reduce(0, owner, (flat @ rv.T).max(1).values, "amax")
        mean = job_mean @ torch.nn.functional.normalize(rv.mean(0), dim=0)
        out[r["id"]] = {j["id"]: [round(b, 5), round(m, 5)] for j, b, m in zip(jobs, best.tolist(), mean.tolist())}
    (OUT / f"emb_{name}.json").write_text(json.dumps(out))
    print(f"  emb_{name}: {time.time() - t0:.0f}s", flush=True)
    del model
    torch.cuda.empty_cache()


def cross(name, resumes, texts, pairs):
    t0 = time.time()
    model = CrossEncoder(CROSS[name], device="cuda", max_length=512, model_kwargs={"torch_dtype": torch.float16})
    by_id = {r["id"]: r for r in resumes}
    items, owner = [], []
    for rid, jid in pairs:
        for rc in chunks(by_id[rid]["text"]):
            for jc in chunks(texts[jid]):
                items.append((rc, jc))
                owner.append((rid, jid))
    scores = model.predict(items, batch_size=32, show_progress_bar=False, activation_fn=torch.nn.Identity())
    out = {}
    for (rid, jid), s in zip(owner, scores):
        d = out.setdefault(rid, {})
        d[jid] = max(d.get(jid, -1e9), float(s))
    (OUT / f"cross_{name}.json").write_text(json.dumps(out))
    print(f"  cross_{name}: {time.time() - t0:.0f}s ({len(items)} chunk pairs)", flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cross", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    resumes, jobs, texts = load()
    only = set(filter(None, args.only.split(",")))
    print(torch.cuda.get_device_name(0), len(resumes), "résumés ×", len(jobs), "jobs", flush=True)
    if args.cross:
        labels = json.loads((EVAL / "labels.json").read_text(encoding="utf-8"))
        pairs = [(x["resume_id"], x["job_id"]) for x in labels]
        for name in CROSS:
            if (not only or name in only) and not (OUT / f"cross_{name}.json").exists():
                try:
                    cross(name, resumes, texts, pairs)
                except Exception as e:                       # a model that won't load shouldn't stop the rest
                    print(f"  cross_{name}: FAILED {type(e).__name__}: {str(e)[:150]}", flush=True)
        return
    for name in EMB:
        if (not only or name in only) and not (OUT / f"emb_{name}.json").exists():
            try:
                embeddings(name, resumes, jobs, texts)
            except Exception as e:
                print(f"  emb_{name}: FAILED {type(e).__name__}: {str(e)[:150]}", flush=True)


if __name__ == "__main__":
    main()
