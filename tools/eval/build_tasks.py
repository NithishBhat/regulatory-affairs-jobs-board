"""Build labeling tasks for the general eval set (TREC-style pooling), on the GPU.

    C:\\Users\\nithi\\venvs\\ra-qa-jobboard\\Scripts\\python.exe tools/eval/build_tasks.py

Input:  cache/general_eval/resumes.json [{id, source, family, text}], site/jobs.json, cache/job_texts.json,
        cache/general_eval/keyword_scores.json (tools/eval/keyword_scores.mjs).
Output: cache/general_eval/tasks/<resume>.json {resume, jobs: [{job_id, title, company, text}]}.
Each résumé's pool is the union of the top jobs from three different rankers (keyword; gte-small best-chunk;
bge-large best-chunk) plus random jobs, so the labels don't favour any one method and include clear negatives.
"""
import json
import random
import re
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "cache" / "general_eval"
TOP, RANDOM, MAX_WORDS, CHUNK = 14, 12, 700, 180
EMAIL = re.compile(r"\S+@\S+")
PHONE = re.compile(r"(\+?\d[\d\s().-]{8,}\d)")
URL = re.compile(r"https?://\S+|www\.\S+")


def scrub(t):
    return URL.sub("[url]", PHONE.sub("[phone]", EMAIL.sub("[email]", t)))


def chunks(t):
    w = t.split()
    return [" ".join(w[i:i + CHUNK]) for i in range(0, len(w), CHUNK)] or [""]


def maxsim(model_id, resumes, jobs, texts, qprefix=""):
    model = SentenceTransformer(model_id, device="cuda", model_kwargs={"torch_dtype": torch.float16})
    model.max_seq_length = 512
    enc = lambda xs: model.encode(xs, batch_size=32, normalize_embeddings=True, convert_to_tensor=True).float()
    job_chunks = [chunks(texts[j["id"]]) for j in jobs]
    flat = enc([c for jc in job_chunks for c in jc])
    owner = torch.tensor([i for i, jc in enumerate(job_chunks) for _ in jc], device="cuda")
    out = {}
    for r in resumes:
        rv = enc([qprefix + c for c in chunks(r["text"])])
        best = (flat @ rv.T).max(1).values                     # per job chunk
        per_job = torch.full((len(jobs),), -1.0, device="cuda").scatter_reduce(0, owner, best, "amax")
        out[r["id"]] = dict(zip([j["id"] for j in jobs], per_job.tolist()))
    del model
    torch.cuda.empty_cache()
    return out


def main():
    resumes = json.loads((EVAL / "resumes.json").read_text(encoding="utf-8"))
    jobs = json.loads((ROOT / "site" / "jobs.json").read_text(encoding="utf-8"))["jobs"]
    texts = {x["id"]: x["text"] for x in json.loads((ROOT / "cache" / "job_texts.json").read_text(encoding="utf-8"))}
    jobs = [j for j in jobs if j["id"] in texts]
    kw = json.loads((EVAL / "keyword_scores.json").read_text(encoding="utf-8"))
    gte = maxsim("thenlper/gte-small", resumes, jobs, texts)
    bge = maxsim("BAAI/bge-large-en-v1.5", resumes, jobs, texts, "Represent this sentence for searching relevant passages: ")
    by_id = {j["id"]: j for j in jobs}
    (EVAL / "tasks").mkdir(parents=True, exist_ok=True)
    rnd = random.Random(13)
    for r in resumes:
        ranked = lambda s: sorted(s, key=s.get, reverse=True)
        pool = []
        for lst in (ranked(kw[r["id"]]), ranked(gte[r["id"]]), ranked(bge[r["id"]])):
            pool += [j for j in lst[:TOP] if j not in pool]
        rest = [j["id"] for j in jobs if j["id"] not in pool]
        pool += rnd.sample(rest, RANDOM)
        rnd.shuffle(pool)
        task = {"resume_id": r["id"], "resume": scrub(r["text"]), "jobs": [
            {"job_id": j, "title": by_id[j]["title"], "company": by_id[j]["company"],
             "text": " ".join(texts[j].split()[:MAX_WORDS])} for j in pool]}
        (EVAL / "tasks" / f"{r['id'].replace(':', '_')}.json").write_text(json.dumps(task, ensure_ascii=False), encoding="utf-8")
    print(len(resumes), "tasks,", sum(1 for _ in (EVAL / "tasks").glob("*.json")), "files; ~",
          TOP * 3 + RANDOM, "jobs max per résumé")


if __name__ == "__main__":
    main()
