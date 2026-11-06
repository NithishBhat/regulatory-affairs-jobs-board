"""JobBERT-v2/v3 have two heads (Router): 'anchor' (job titles) and 'positive' (skills). Score résumé × job
best-chunk similarity with each head combination (résumé head / job head).

    C:\\Users\\nithi\\venvs\\ra-qa-jobboard\\Scripts\\python.exe -I tools/eval/routes.py

Writes scores/emb_<model>_<rhead>-<jhead>.json and tools/results/routes.md.
"""
import json
import sys
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import EVAL, RESULTS, ROOT, blend, bootstrap_diff, load_labels, per_resume, summary, zscore_all  # noqa: E402
from finetune import chunks  # noqa: E402

MODELS = {"jobbert-v2": "TechWolf/JobBERT-v2", "jobbert-v3": "TechWolf/JobBERT-v3"}


def main():
    labels = load_labels("labels")
    resumes = [r for r in json.loads((EVAL / "resumes.json").read_text(encoding="utf-8")) if r["id"] in labels]
    kind = {r["id"]: r["kind"] for r in resumes}
    real = {r for r in labels if kind.get(r) == "real"}
    jobs = json.loads((ROOT / "site" / "jobs.json").read_text(encoding="utf-8"))["jobs"]
    texts = {x["id"]: x["text"] for x in json.loads((ROOT / "cache" / "job_texts.json").read_text(encoding="utf-8"))}
    ids = [j["id"] for j in jobs if j["id"] in texts]
    load = lambda n: json.loads((EVAL / "scores" / n).read_text(encoding="utf-8"))
    kw, lex, sem, site = (load(f"site_{n}.json") for n in ("keyword", "lexical", "semantic", "final"))
    fmt = lambda t: f"{t[0]:.3f} / {t[1]:.3f} / {t[2]:.2f}"
    lines = ["# JobBERT heads — résumé head / job head, best 180-word chunk", "",
             "Blend = keyword 0.3 + TF-IDF 0.3 + gte-base 0.2 + this 0.2. Δρ vs the site.", "",
             "| Model | heads | alone (all) | alone (real) | blend (all) | Δρ vs site (95% CI) |", "|---|---|---|---|---|---|"]
    for name, mid in MODELS.items():
        model = SentenceTransformer(mid, device="cuda", model_kwargs={"torch_dtype": torch.float16})
        model.max_seq_length = 512
        enc = lambda xs, task: model.encode(xs, batch_size=32, normalize_embeddings=True, convert_to_tensor=True,
                                            task=task).float()
        jc = [chunks(texts[i]) for i in ids]
        flat = [c for x in jc for c in x]
        owner = torch.tensor([i for i, x in enumerate(jc) for _ in x], device="cuda")
        J = {h: enc(flat, h) for h in ("anchor", "positive")}
        for rh, jh in (("anchor", "anchor"), ("positive", "positive"), ("anchor", "positive"), ("positive", "anchor")):
            out = {}
            for r in resumes:
                best = (J[jh] @ enc(chunks(r["text"]), rh).T).max(1).values
                per = torch.full((len(ids),), -1.0, device="cuda").scatter_reduce(0, owner, best, "amax")
                out[r["id"]] = dict(zip(ids, per.tolist()))
            (EVAL / "scores" / f"emb_{name}_{rh}-{jh}.json").write_text(json.dumps(out))
            b = blend([zscore_all(kw), zscore_all(lex), zscore_all(sem), zscore_all(out)], [0.3, 0.3, 0.2, 0.2])
            a = per_resume(out, labels)
            d = bootstrap_diff(b, site, labels)
            lines.append(f"| {name} | {rh} / {jh} | {fmt(summary(a))} | {fmt(summary(a, real))} | "
                         f"{fmt(summary(per_resume(b, labels)))} | {d[0]:+.3f} ({d[1]:+.3f} … {d[2]:+.3f}) |")
        del model
        torch.cuda.empty_cache()
    (RESULTS / "routes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
