"""Check the site's own scores (tools/eval/site_scores.mjs) against the labels, vs the previous site.

    node tools/eval/site_scores.mjs
    C:\\Users\\nithi\\venvs\\ra-qa-jobboard\\Scripts\\python.exe -I tools/eval/eval_site.py

"Previous site" = keyword + gte-small best chunk, 50/50 (scores/emb_gte-small.json). Writes
tools/results/site_eval.md. Also prints the CV-picked weights for keyword + TF-IDF (the site's fallback
before the model loads).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (EVAL, RESULTS, blend, bootstrap_diff, cv_blend, load_labels, per_resume,  # noqa: E402
                      summary, zscore_all)

S = EVAL / "scores"


def load(name):
    return json.loads((S / name).read_text(encoding="utf-8"))


def main():
    labels = load_labels("labels")
    kind = {r["id"]: r["kind"] for r in json.loads((EVAL / "resumes.json").read_text(encoding="utf-8"))}
    real = {r for r in labels if kind.get(r) == "real"}
    syn = {r for r in labels if kind.get(r) == "synthetic"}
    gte_small = {r: {j: v[0] for j, v in x.items()} for r, x in load("emb_gte-small.json").items()}
    kw = load("site_keyword.json")
    old = blend([zscore_all(kw), zscore_all(gte_small)], [0.5, 0.5])
    _, ws = cv_blend(kw, load("site_lexical.json"), labels)
    rows = [("Previous site: keyword + gte-small 50/50", old),
            ("Keyword only", kw),
            ("New site, before the model loads: keyword + TF-IDF (shown scores)", load("site_final_no_semantic.json"))]
    if (S / "site_final.json").exists():
        rows.append(("New site: keyword + TF-IDF + gte-base (shown scores)", load("site_final.json")))
    fmt = lambda t: " / ".join(f"{x:.3f}" if i < 2 else f"{x:.2f}" for i, x in enumerate(t))
    lines = ["# Site eval: the site's own code on the 40-résumé set", "",
             "Per-résumé ρ / nDCG@10 / P@10. Scores are the integers visitors see (site/match.js).", "",
             "| Strategy | all | real | synthetic | Δρ vs previous (95% CI) |", "|---|---|---|---|---|"]
    for name, sc in rows:
        pr = per_resume(sc, labels)
        d = bootstrap_diff(sc, old, labels) if sc is not old else None
        ci = f"{d[0]:+.3f} ({d[1]:+.3f} … {d[2]:+.3f})" if d else "—"
        lines.append(f"| {name} | {fmt(summary(pr))} | {fmt(summary(pr, real))} | {fmt(summary(pr, syn))} | {ci} |")
    lines += ["", f"CV weight on keyword for keyword + TF-IDF, per fold: {ws}"]
    (RESULTS / "site_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
