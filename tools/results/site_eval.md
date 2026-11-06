# Site eval: the site's own code on the 40-résumé set

Per-résumé ρ / nDCG@10 / P@10. Scores are the integers visitors see (site/match.js).

| Strategy | all | real | synthetic | Δρ vs previous (95% CI) |
|---|---|---|---|---|
| Previous site: keyword + gte-small 50/50 | 0.597 / 0.528 / 0.30 | 0.526 / 0.467 / 0.26 | 0.703 / 0.620 / 0.36 | — |
| Keyword only | 0.415 / 0.389 / 0.26 | 0.268 / 0.277 / 0.18 | 0.637 / 0.558 / 0.38 | -0.181 (-0.245 … -0.116) |
| New site, before the model loads: keyword + TF-IDF (shown scores) | 0.647 / 0.612 / 0.35 | 0.589 / 0.572 / 0.32 | 0.733 / 0.673 / 0.38 | +0.050 (+0.015 … +0.092) |
| New site: keyword + TF-IDF + gte-base (shown scores) | 0.685 / 0.647 / 0.36 | 0.637 / 0.619 / 0.34 | 0.758 / 0.688 / 0.39 | +0.089 (+0.060 … +0.122) |

CV weight on keyword for keyword + TF-IDF, per fold: [0.3, 0.4, 0.3, 0.3, 0.3]
