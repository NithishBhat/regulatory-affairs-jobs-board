# Experience-gap penalty after the blend — 40 résumés

ρ / nDCG@10 / P@10. Params = (points per missing year, cap, weight when years come from the title, director under-8-years penalty). Only résumés whose years are known (dates or a stated number) are penalised.

| Variant | all | real | Δρ vs site (95% CI) |
|---|---|---|---|
| Site now (no penalty) | 0.677 / 0.637 / 0.35 | 0.636 / 0.615 / 0.33 | +0.000 (+0.000 … +0.000) |
| Penalty, CV-picked per fold | 0.681 / 0.637 / 0.35 | 0.637 / 0.618 / 0.34 | +0.003 (-0.010 … +0.015) |
| Penalty (15, 10, 0.5, 15) (picked on all 40; optimistic) | 0.687 / 0.648 / 0.36 | 0.638 / 0.618 / 0.34 | +0.009 (-0.001 … +0.020) |

CV picks per fold: [(15, 10, 0.5, 0), (15, 10, 0.5, 15), (15, 10, 0.5, 0), (15, 20, 0, 0), (15, 10, 0.5, 30)]
