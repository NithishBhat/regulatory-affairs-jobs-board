# Résumé→job ranking strategies — 2026-11-04

398 jobs scored by an LLM against one RA/QA résumé (rubric 0–100; ≥70 = good). ρ = Spearman with those
scores over all jobs; nDCG@20 / P@20 = quality of the top 20. Blend weights chosen on one half of the jobs,
scored on the other. Single résumé + LLM labels: treat differences under ~0.03 ρ as noise.

| # | Strategy | From | ρ | nDCG@20 | P@20 |
|---|---|---|---|---|---|
| 1 | Keyword + e5-large max-sim (GPU) (w=0.6/0.6) | large-model ceiling | 0.658 | 0.632 | 0.90 |
| 2 | Keyword + bge-large max-sim (GPU) (w=0.6/0.6) | large-model ceiling | 0.654 | 0.554 | 0.95 |
| 3 | Keyword + gte-large max-sim (GPU) (w=0.5/0.6) | large-model ceiling | 0.647 | 0.614 | 0.85 |
| 4 | Keyword + gte-small max-sim, w=0.5 (current site) | this project | 0.624 | 0.552 | 0.80 |
| 5 | Current site → rerank top 60 by bge-reranker-v2-m3 | retrieve-then-rerank posts | 0.618 | 0.405 | 0.60 |
| 6 | Current site → rerank top 60 by cross-encoder | retrieve-then-rerank posts | 0.616 | 0.398 | 0.70 |
| 7 | Current site → rerank top 60 by bge-reranker-base | retrieve-then-rerank posts | 0.615 | 0.308 | 0.65 |
| 8 | e5-large: max chunk similarity (GPU) | large-model ceiling | 0.608 | 0.518 | 0.80 |
| 9 | Keyword + gte-small requirements max-sim (w=0.6/0.5) | this project + requirement focus | 0.607 | 0.635 | 0.90 |
| 10 | Keyword + gte-base max-sim (w=0.5/0.5) | this project, bigger models | 0.598 | 0.671 | 0.80 |
| 11 | bge-large: max chunk similarity (GPU) | large-model ceiling | 0.597 | 0.491 | 0.80 |
| 12 | Keyword + bge-base max-sim (w=0.5/0.5) | this project, bigger models | 0.595 | 0.533 | 0.85 |
| 13 | gte-large: max chunk similarity (GPU) | large-model ceiling | 0.581 | 0.489 | 0.70 |
| 14 | Keyword + bge-small max-sim (w=0.4/0.4) | this project, bigger models | 0.581 | 0.544 | 0.75 |
| 15 | mpnet-base: max chunk similarity (GPU) | large-model ceiling | 0.578 | 0.380 | 0.60 |
| 16 | Keyword + e5-base max-sim (w=0.4/0.5) | this project, bigger models | 0.563 | 0.418 | 0.65 |
| 17 | RRF(keyword, BM25, gte-small max-sim) | reciprocal rank fusion posts (k=60) | 0.548 | 0.446 | 0.75 |
| 18 | RRF(keyword, gte-small max-sim, cross-encoder) | RRF + reranker | 0.544 | 0.546 | 0.75 |
| 19 | gte-small: max chunk similarity | chunked retrieval | 0.537 | 0.421 | 0.75 |
| 20 | Keyword + cross-encoder (w=0.2/0.2) | hybrid + reranker | 0.526 | 0.572 | 0.70 |
| 21 | Keyword + bge-reranker-v2-m3 (w=0.3/0.3) | hybrid + reranker | 0.524 | 0.571 | 0.75 |
| 22 | BM25 + gte-small max-sim (w=0.8/1) | hybrid search posts | 0.522 | 0.417 | 0.75 |
| 23 | gte-small max-sim, requirements section only | JobMatch-AI requirement focus | 0.515 | 0.513 | 0.90 |
| 24 | bge-base: max chunk similarity | chunked retrieval | 0.510 | 0.420 | 0.65 |
| 25 | Keyword vocab (current site, layer 1) | this project | 0.504 | 0.485 | 0.65 |
| 26 | Keyword + bge-reranker-base (w=0.1/0) | hybrid + reranker | 0.504 | 0.472 | 0.65 |
| 27 | gte-base: max chunk similarity | chunked retrieval | 0.503 | 0.450 | 0.70 |
| 28 | TF-IDF + MiniLM mean (w=0.1/0.3) | lisan-5/resume-job-matching hybrid | 0.478 | 0.466 | 0.75 |
| 29 | e5-large: mean embedding cosine (GPU) | large-model ceiling | 0.471 | 0.454 | 0.70 |
| 30 | mpnet-base: mean embedding cosine (GPU) | most-used SBERT in Medium posts | 0.468 | 0.330 | 0.75 |
| 31 | TF-IDF cosine, uni+bigrams | classic Medium tutorials | 0.465 | 0.471 | 0.80 |
| 32 | MiniLM-L6: max chunk similarity | chunked retrieval | 0.455 | 0.273 | 0.45 |
| 33 | e5-base: max chunk similarity | chunked retrieval | 0.450 | 0.327 | 0.55 |
| 34 | bge-small: max chunk similarity | chunked retrieval | 0.448 | 0.385 | 0.55 |
| 35 | gte-small: mean embedding cosine | SBERT-cosine repos | 0.445 | 0.459 | 0.60 |
| 36 | bge-large: mean embedding cosine (GPU) | large-model ceiling | 0.442 | 0.419 | 0.70 |
| 37 | TF-IDF cosine, unigrams | classic Medium tutorials | 0.438 | 0.469 | 0.80 |
| 38 | gte-large: mean embedding cosine (GPU) | large-model ceiling | 0.434 | 0.481 | 0.70 |
| 39 | gte-base: mean embedding cosine | SBERT-cosine repos | 0.411 | 0.445 | 0.70 |
| 40 | bge-base: mean embedding cosine | SBERT-cosine repos | 0.388 | 0.359 | 0.70 |
| 41 | Skill Jaccard (vocab terms) | skill-extractor repos (SkillTool etc.) | 0.375 | 0.380 | 0.65 |
| 42 | MiniLM-L6: mean embedding cosine | SBERT-cosine repos | 0.373 | 0.345 | 0.60 |
| 43 | bge-small: mean embedding cosine | SBERT-cosine repos | 0.355 | 0.405 | 0.60 |
| 44 | e5-base: mean embedding cosine | SBERT-cosine repos | 0.353 | 0.266 | 0.60 |
| 45 | Cross-encoder bge-reranker-v2-m3 (all jobs, GPU) | reranker posts | 0.275 | 0.370 | 0.65 |
| 46 | BM25 (résumé as query) | rank_bm25 / HF resume-screener demo | 0.275 | 0.333 | 0.65 |
| 47 | Cross-encoder ms-marco-MiniLM (all jobs) | HF resume-screener demo | 0.240 | 0.296 | 0.60 |
| 48 | Skill coverage (job terms found) | Resume-Matcher keyword % | 0.191 | 0.528 | 0.55 |
| 49 | Cross-encoder bge-reranker-base (all jobs, GPU) | reranker posts | 0.091 | 0.181 | 0.40 |

Approx. q8 model download for in-browser use (MB): MiniLM-L6 23, bge-small 34, gte-small 34, bge-base 110, e5-base 110, gte-base 110, ms-marco-MiniLM cross-encoder 23, bge-reranker-base ~280, bge-reranker-v2-m3 ~570; *-large ~335 (GPU-only rows are not browser-practical).

Not run: LLM-based extraction / judging (JobMatch-AI two-pass, ConFit v2 hypothetical résumé) — need an LLM
per job; SPLADE (no transformers.js port); fine-tuned models (ConFit, conSultantBERT) — need labeled pairs at scale.
