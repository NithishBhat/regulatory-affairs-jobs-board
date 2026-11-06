# Score calibration — blended score -> expected rubric score

5-fold CV over résumés on the labeled pairs (labels: tools/eval/rubric.md, 85+ strong, 70–84 good, 50–69 partial).

## After the AI model loads (keyword + TF-IDF + embeddings)

| Shown score | mean abs. error vs label | tier agrees with label |
|---|---|---|
| Before (blended, capped at 100; old cut-offs 68 / 50) | 17.5 | 59% |
| Calibrated (rubric tiers 85 / 70 / 50) | 13.2 | 68% |

Held-out pairs shown at 70+: 92; label 70+ for 67% (mean label 74).

Table (x = blended score, y = shown score):

| x | -6 | 20 | 27 | 32 | 35 | 38 | 41 | 43 | 45 | 47 | 49 | 51 | 53 | 54 | 57 | 59 | 61 | 63 | 65 | 68 | 71 | 74 | 79 | 87 | 132 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| y | 4 | 13 | 18 | 18 | 20 | 25 | 29 | 29 | 29 | 34 | 34 | 37 | 38 | 44 | 44 | 44 | 47 | 52 | 52 | 54 | 58 | 58 | 62 | 72 | 90 |

## Before it loads (keyword + TF-IDF)

| Shown score | mean abs. error vs label | tier agrees with label |
|---|---|---|
| Before (blended, capped at 100; old cut-offs 68 / 50) | 17.4 | 62% |
| Calibrated (rubric tiers 85 / 70 / 50) | 13.6 | 68% |

Held-out pairs shown at 70+: 49; label 70+ for 71% (mean label 74).

Table (x = blended score, y = shown score):

| x | -2 | 21 | 25 | 29 | 32 | 35 | 38 | 40 | 42 | 44 | 46 | 48 | 50 | 52 | 55 | 57 | 59 | 62 | 64 | 68 | 72 | 76 | 82 | 92 | 165 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| y | 5 | 13 | 17 | 20 | 25 | 25 | 25 | 28 | 31 | 31 | 32 | 35 | 38 | 43 | 48 | 48 | 48 | 48 | 52 | 52 | 58 | 58 | 60 | 69 | 91 |
