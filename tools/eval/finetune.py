"""Fine-tune an embedder on the labeled pairs, 5-fold CV over résumés (GPU). Serving cost is unchanged: the
fine-tuned model is the same size as the base one.

    C:\\Users\\nithi\\venvs\\ra-qa-jobboard\\Scripts\\python.exe -I tools/eval/finetune.py [--model gte-base] [--epochs 2]

Each fold trains on 32 résumés' labeled pairs (CoSENT loss on score/100; résumé and job text as the raters saw
them, truncated to 512 tokens) and scores the 8 held-out résumés against every job with the site's method
(best 180-word chunk pair). So every résumé is scored by a model that never saw it. Writes
cache/general_eval/scores/ft_<model>_e<epochs>.json and appends to tools/results/finetune.md.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.losses import CoSENTLoss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import EVAL, RESULTS, ROOT, blend, bootstrap_diff, load_labels, per_resume, summary, zscore_all  # noqa: E402

MODELS = {"gte-base": "thenlper/gte-base", "jobbert-v2": "TechWolf/JobBERT-v2", "bge-base": "BAAI/bge-base-en-v1.5"}
SCRATCH = ROOT / "cache" / "finetune"


def chunks(t, n=180):
    w = str(t or "").split()
    return [" ".join(w[i:i + n]) for i in range(0, len(w), n)] or [""]


def score(model, resumes, ids, docs):
    enc = lambda xs: model.encode(xs, batch_size=32, normalize_embeddings=True, convert_to_tensor=True).float()
    jc = [chunks(d) for d in docs]
    J = enc([c for x in jc for c in x])
    owner = torch.tensor([i for i, x in enumerate(jc) for _ in x], device="cuda")
    out = {}
    for r in resumes:
        best = (J @ enc(chunks(r["text"])).T).max(1).values
        per = torch.full((len(docs),), -1.0, device="cuda").scatter_reduce(0, owner, best, "amax")
        out[r["id"]] = dict(zip(ids, per.tolist()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gte-base", choices=list(MODELS))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--strict", action="store_true",
                    help="also drop training pairs whose job is labeled for a held-out résumé (unseen jobs, like new postings)")
    args = ap.parse_args()
    labels = load_labels("labels")
    resumes = [r for r in json.loads((EVAL / "resumes.json").read_text(encoding="utf-8")) if r["id"] in labels]
    kind = {r["id"]: r["kind"] for r in resumes}
    real = {r for r in labels if kind.get(r) == "real"}
    jobs = json.loads((ROOT / "site" / "jobs.json").read_text(encoding="utf-8"))["jobs"]
    texts = {x["id"]: x["text"] for x in json.loads((ROOT / "cache" / "job_texts.json").read_text(encoding="utf-8"))}
    ids = [j["id"] for j in jobs if j["id"] in texts]
    docs = [texts[i] for i in ids]
    tasks = {}
    for f in (EVAL / "tasks").glob("*.json"):
        t = json.loads(f.read_text(encoding="utf-8"))
        tasks[t["resume_id"]] = t

    rids = sorted(labels)
    random.Random(3).shuffle(rids)
    out, t0 = {}, time.time()
    for k in range(5):
        test = set(rids[k::5])
        rows = {"anchor": [], "positive": [], "score": []}
        held_jobs = {j for r in test for j in labels[r]} if args.strict else set()
        for rid in rids:
            if rid in test:
                continue
            t = tasks[rid]
            for j in t["jobs"]:
                if j["job_id"] in labels[rid] and j["job_id"] not in held_jobs:
                    rows["anchor"].append(t["resume"])
                    rows["positive"].append(f"{j['title']}. {j['text']}")
                    rows["score"].append(labels[rid][j["job_id"]] / 100)
        model = SentenceTransformer(MODELS[args.model], device="cuda", model_kwargs={"torch_dtype": torch.float32})
        model.max_seq_length = 512
        targs = SentenceTransformerTrainingArguments(
            output_dir=str(SCRATCH), num_train_epochs=args.epochs, per_device_train_batch_size=args.batch,
            learning_rate=args.lr, warmup_ratio=0.1, fp16=True, save_strategy="no", logging_steps=1000,
            report_to=[], seed=k, gradient_checkpointing=True,
            # JobBERT has a Router with 'anchor' (titles) and 'positive' (skills) heads; both sides use the
            # default 'anchor' head, which is what inference uses (tools/results/routes.md).
            **({"router_mapping": {"anchor": "anchor", "positive": "anchor"}}
               if any(type(m).__name__ == "Router" for m in model.modules()) else {}))
        trainer = SentenceTransformerTrainer(model=model, args=targs, train_dataset=Dataset.from_dict(rows),
                                             loss=CoSENTLoss(model))
        trainer.train()
        model.eval()
        out.update(score(model, [r for r in resumes if r["id"] in test], ids, docs))
        print(f"fold {k}: {len(rows['score'])} pairs, {time.time() - t0:.0f}s", flush=True)
        del model, trainer
        torch.cuda.empty_cache()

    name = f"ft_{args.model}_e{args.epochs}" + ("_strict" if args.strict else "")
    (EVAL / "scores" / f"{name}.json").write_text(json.dumps(out))
    load = lambda n: json.loads((EVAL / "scores" / n).read_text(encoding="utf-8"))
    kw, lex, site = load("site_keyword.json"), load("site_lexical.json"), load("site_final.json")
    base = {r: {j: v[0] for j, v in x.items()} for r, x in load(f"emb_{args.model}.json").items()}
    fmt = lambda t: f"{t[0]:.3f} / {t[1]:.3f} / {t[2]:.2f}"
    lines = []
    for label, emb in ((f"{args.model} (base)", base), (f"{args.model} fine-tuned, {args.epochs} epochs, lr {args.lr}"
                                                           + (", strict (held-out jobs too)" if args.strict else ""), out)):
        b = blend([zscore_all(kw), zscore_all(lex), zscore_all(emb)], [0.3, 0.3, 0.4])
        a, bl = per_resume(emb, labels), per_resume(b, labels)
        d = bootstrap_diff(b, site, labels)
        lines.append(f"| {label} | {fmt(summary(a))} | {fmt(summary(a, real))} | {fmt(summary(bl))} | "
                     f"{fmt(summary(bl, real))} | {d[0]:+.3f} ({d[1]:+.3f} … {d[2]:+.3f}) |")
    md = RESULTS / "finetune.md"
    if not md.exists():
        md.write_text("# Fine-tuned embedders — 5-fold CV over résumés\n\nρ / nDCG@10 / P@10. Blend = keyword 0.3 + "
                      "TF-IDF 0.3 + embedding 0.4 (site weights). Δρ vs the site's shown scores.\n\n"
                      "| Model | alone (all) | alone (real) | blend (all) | blend (real) | Δρ vs site (95% CI) |\n"
                      "|---|---|---|---|---|---|\n", encoding="utf-8")
    with md.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
