# Site eval: the site's own code on the 40-résumé set

Per-résumé ρ / nDCG@10 / P@10. Scores are the integers visitors see (site/match.js).

| Strategy | all | real | synthetic | Δρ vs previous (95% CI) |
|---|---|---|---|---|
| Previous site: keyword + gte-small 50/50 | 0.601 / 0.532 / 0.31 | 0.531 / 0.470 / 0.26 | 0.706 / 0.624 / 0.38 | — |
| Keyword only | 0.417 / 0.389 / 0.27 | 0.269 / 0.279 / 0.19 | 0.639 / 0.555 / 0.38 | -0.184 (-0.246 … -0.119) |
| New site, before the model loads: keyword + TF-IDF (shown scores) | 0.641 / 0.610 / 0.35 | 0.589 / 0.582 / 0.33 | 0.718 / 0.652 / 0.38 | +0.040 (+0.006 … +0.079) |
| New site: keyword + TF-IDF + gte-base (shown scores) | 0.683 / 0.646 / 0.36 | 0.645 / 0.628 / 0.35 | 0.740 / 0.674 / 0.39 | +0.082 (+0.052 … +0.117) |

CV weight on keyword for keyword + TF-IDF, per fold: [0.3, 0.3, 0.3, 0.3, 0.3]
