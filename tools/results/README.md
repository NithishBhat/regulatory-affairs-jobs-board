# Ranking experiments

`strategies.md` / `strategies.json` hold 49 strategies scored on 398 jobs that an LLM rated against one
RA/QA résumé (run 2026-11-04). Re-run: `C:\Users\nithi\venvs\ra-qa-jobboard\Scripts\python.exe tools/gpu_scores.py`
(cross-encoders + large models on the GPU), then `node tools/strategies.mjs`. Scores are cached in
`cache/strategy_scores/`; the data (`cache/eval.json`) holds a real résumé, so it stays out of git.

## Takeaways (2026-11-04)
- **The current site setup (keyword + gte-small best-chunk similarity, 50/50) is the best browser-sized
  option**: ρ 0.624, #4 overall. Only the large models beat it: e5-large / bge-large + keyword reach ρ 0.65–0.66,
  a +0.03 gain that is near the noise level, at roughly 10× the download (about 335 MB).
- **Hybrids beat every single method.** Every embedding model did better blended with the keyword layer, which
  suggests the domain vocabulary carries signal the models miss.
- **Best-chunk similarity beats averaging the whole posting** for every model (e.g. gte-small 0.537 vs 0.445).
  Long postings dilute a mean.
- **Embedding only the requirements section** (gte-small) gave the best top-20 of the small models
  (nDCG@20 0.635, P@20 0.90) but a lower overall ρ (0.607). It's worth a second test with more résumés.
- **Cross-encoders / rerankers did badly here** (ms-marco, bge-reranker-base, bge-reranker-v2-m3: ρ 0.09–0.28;
  reranking the top 60 cut nDCG@20 from 0.55 to 0.31–0.41). They are trained on short-query → passage search,
  and a résumé chunk isn't a query.
- **Classic baselines from tutorials are weak.** TF-IDF 0.44–0.47, BM25 0.28, skill Jaccard 0.38, and
  Resume-Matcher-style keyword coverage 0.19 (though its top 20 is decent: nDCG 0.53).
- Caveats: one résumé, LLM labels rather than hiring outcomes, and GPU rows run in fp16 PyTorch rather than the
  site's q8 ONNX.
- Not tested: LLM extraction / judging and the ConFit v2 hypothetical résumé (need an LLM per job, e.g. Bedrock
  batch), SPLADE, and fine-tuned models (need many labeled pairs).
