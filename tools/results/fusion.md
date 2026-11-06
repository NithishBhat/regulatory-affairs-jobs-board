# Fusion methods — 40 résumés

ρ / nDCG@10 / P@10; Δρ vs the site's shown scores.

| Method | all | real | Δρ vs site (95% CI) |
|---|---|---|---|
| Subset search, minmax-sum (CV picks: keyword+tfidf+jobbert-v2+jobbert-v3+gte-base; keyword+tfidf+jobbert-v2+jobbert-v3+bge-base; keyword+tfidf+jobbert-v2+jobbert-v3+bge-base; keyword+tfidf+qwen3-emb-0.6b+jobbert-v3+resume-jd-mpnet; keyword+tfidf+qwen3-emb-0.6b+jobbert-v3+resume-jd-mpnet) | 0.712 / 0.654 / 0.37 | 0.670 / 0.603 / 0.35 | +0.027 (+0.010 … +0.045) |
| Subset search, z-equal (CV picks: keyword+tfidf+jobbert-v2+jobbert-v3+gte-base; keyword+tfidf+jobbert-v2+qwen3-emb-0.6b+jobbert-v3; keyword+tfidf+jobbert-v2+jobbert-v3+bge-base; keyword+tfidf+jobbert-v2+jobbert-v3+gte-base; keyword+tfidf+jobbert-v2+jobbert-v3+resume-jd-mpnet) | 0.708 / 0.662 / 0.37 | 0.683 / 0.629 / 0.34 | +0.023 (+0.007 … +0.040) |
| keyword + TF-IDF + jobbert-v2: z-equal | 0.690 / 0.630 / 0.36 | 0.640 / 0.580 / 0.33 | +0.005 (-0.014 … +0.026) |
| keyword + TF-IDF + gte-base: z-equal | 0.686 / 0.645 / 0.36 | 0.645 / 0.617 / 0.35 | +0.001 (-0.006 … +0.009) |
| Site (z-weighted 0.3/0.3/0.4, gte-base in browser) | 0.685 / 0.649 / 0.37 | 0.646 / 0.627 / 0.35 | +0.000 (+0.000 … +0.000) |
| keyword + TF-IDF + jobbert-v2: minmax-geo | 0.683 / 0.621 / 0.37 | 0.631 / 0.563 / 0.33 | -0.002 (-0.020 … +0.019) |
| keyword + TF-IDF + jobbert-v2: minmax-sum | 0.682 / 0.637 / 0.38 | 0.623 / 0.585 / 0.35 | -0.002 (-0.027 … +0.024) |
| keyword + TF-IDF + arctic-l-v2: z-equal | 0.679 / 0.647 / 0.35 | 0.631 / 0.615 / 0.32 | -0.006 (-0.020 … +0.007) |
| keyword + TF-IDF + qwen3-emb-0.6b: z-equal | 0.677 / 0.643 / 0.37 | 0.621 / 0.596 / 0.33 | -0.007 (-0.027 … +0.012) |
| keyword + TF-IDF + gte-base: minmax-sum | 0.677 / 0.633 / 0.36 | 0.623 / 0.592 / 0.34 | -0.007 (-0.023 … +0.008) |
| keyword + TF-IDF + arctic-l-v2: minmax-geo | 0.677 / 0.638 / 0.35 | 0.628 / 0.598 / 0.32 | -0.008 (-0.023 … +0.008) |
| keyword + TF-IDF + qwen3-emb-0.6b: minmax-geo | 0.677 / 0.623 / 0.37 | 0.620 / 0.569 / 0.34 | -0.008 (-0.028 … +0.012) |
| keyword + TF-IDF + gte-base: minmax-geo | 0.676 / 0.629 / 0.36 | 0.630 / 0.596 / 0.35 | -0.008 (-0.022 … +0.004) |
| keyword + TF-IDF + arctic-l-v2: minmax-sum | 0.676 / 0.647 / 0.36 | 0.618 / 0.612 / 0.33 | -0.009 (-0.029 … +0.011) |
| keyword + TF-IDF + qwen3-emb-0.6b: minmax-sum | 0.671 / 0.636 / 0.37 | 0.603 / 0.579 / 0.33 | -0.014 (-0.039 … +0.012) |
| keyword + TF-IDF + gte-base: rrf | 0.659 / 0.637 / 0.36 | 0.596 / 0.601 / 0.33 | -0.026 (-0.045 … -0.007) |
| keyword + TF-IDF + arctic-l-v2: rrf | 0.658 / 0.655 / 0.36 | 0.593 / 0.612 / 0.34 | -0.027 (-0.051 … -0.004) |
| keyword + TF-IDF + jobbert-v2: rrf | 0.655 / 0.663 / 0.37 | 0.581 / 0.594 / 0.34 | -0.029 (-0.057 … -0.001) |
| keyword + TF-IDF + qwen3-emb-0.6b: rrf | 0.640 / 0.636 / 0.36 | 0.555 / 0.565 / 0.31 | -0.045 (-0.078 … -0.011) |
