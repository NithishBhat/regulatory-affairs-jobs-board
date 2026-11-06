"""GPU (CUDA) scores for the strategy comparison: cross-encoders and large embedding models.

    C:\\Users\\nithi\\venvs\\ra-qa-jobboard\\Scripts\\python.exe tools/gpu_scores.py [cache/eval.json]

Writes cache/strategy_scores/<name>.json in the same format tools/strategies.mjs reads
({ids, value, secs}); existing files are kept. Chunking matches site/match.js (180-word chunks).
"""
import json
import sys
import time
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "cache" / "strategy_scores"
CHUNK_WORDS = 180

CROSS = {
    "cross_msmarco_all": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross_bge_reranker_all": "BAAI/bge-reranker-base",
    "cross_bge_reranker_v2m3_all": "BAAI/bge-reranker-v2-m3",
}
# name -> (model, query prefix, passage prefix)
EMB = {
    "emb_mpnet-base": ("sentence-transformers/all-mpnet-base-v2", "", ""),
    "emb_bge-large": ("BAAI/bge-large-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
    "emb_e5-large": ("intfloat/e5-large-v2", "query: ", "passage: "),
    "emb_gte-large": ("thenlper/gte-large", "", ""),
}


def chunks(text):
    w = str(text or "").split()
    return [" ".join(w[i:i + CHUNK_WORDS]) for i in range(0, len(w), CHUNK_WORDS)] or [""]


def save(name, ids, value, t0):
    (OUT / f"{name}.json").write_text(json.dumps({"ids": ids, "value": value, "secs": time.time() - t0}))
    print(f"  {name}: {time.time() - t0:.0f}s", flush=True)


def main():
    data = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "cache" / "eval.json").read_text(encoding="utf-8"))
    jobs, ids = data["jobs"], [j["id"] for j in data["jobs"]]
    OUT.mkdir(parents=True, exist_ok=True)
    rc = chunks(data["resume"])
    job_chunks = [chunks(f"{j['title']}. {j['description']}") for j in jobs]
    print(torch.cuda.get_device_name(0), f"{len(jobs)} jobs", flush=True)

    for name, model_id in CROSS.items():
        if (OUT / f"{name}.json").exists():
            continue
        t0 = time.time()
        model = CrossEncoder(model_id, device="cuda", max_length=512, model_kwargs={"torch_dtype": torch.float16})
        pairs, owner = [], []
        for i, jc in enumerate(job_chunks):
            for r in rc:
                for c in jc:
                    pairs.append((r, c))
                    owner.append(i)
        scores = model.predict(pairs, batch_size=32, show_progress_bar=False, activation_fn=torch.nn.Identity())
        best = [-1e9] * len(jobs)
        for i, s in zip(owner, scores):
            best[i] = max(best[i], float(s))
        save(name, ids, best, t0)
        del model
        torch.cuda.empty_cache()

    for name, (model_id, qp, pp) in EMB.items():
        if (OUT / f"{name}.json").exists():
            continue
        t0 = time.time()
        model = SentenceTransformer(model_id, device="cuda", model_kwargs={"torch_dtype": torch.float16})
        model.max_seq_length = 512
        enc = lambda texts: model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_tensor=True).float()
        r = enc([qp + t for t in rc])
        rm = torch.nn.functional.normalize(r.mean(0), dim=0)
        flat = [pp + c for jc in job_chunks for c in jc]
        e = enc(flat)
        mean, maxsim, k = [], [], 0
        for jc in job_chunks:
            v = e[k:k + len(jc)]
            k += len(jc)
            mean.append(float(rm @ torch.nn.functional.normalize(v.mean(0), dim=0)))
            maxsim.append(float((v @ r.T).max()))
        save(name, ids, {"mean": mean, "maxsim": maxsim}, t0)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
