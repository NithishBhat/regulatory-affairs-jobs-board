# RA/QA Job Match

A public job board for Regulatory Affairs / Quality roles. Visitors drop in a résumé and get strong and
weak matches plus tips for widening their search. The résumé is parsed and matched **in the browser**
and never uploaded.

## Run locally
```
py refresh.py                                  # rebuild the job data from the pipeline, test, show changes
py -m http.server 8765 --directory site        # open http://localhost:8765
```

## Refreshing the data
`refresh.py` runs only when you start it; nothing is scheduled.
```
py refresh.py                 # rebuild + sanity checks + tests; nothing goes live
py refresh.py --check-links   # also re-check that job links are still open (contacts employer sites)
py refresh.py --publish       # after a good rebuild, commit the data files and push (goes live)
```
It backs up the current data first and puts it back if anything fails: fewer than 100 jobs, a drop of 40% or
more, or a failing test. `--publish` commits only `site/jobs.json`, `lexical.*` and `vectors.*`, never code.
Each run adds a line to `cache/refresh.log`. To run it daily later, a Windows scheduled task after the
pipeline's 5 AM run would call `py refresh.py --publish` from this folder.

## Tests
```
node --test tools/test/match.test.mjs    # matching, TF-IDF, blend, site data files, no non-US jobs
py tools/test/test_export.py             # US-state parsing, non-US filter, required years
```

## Data
`export.py` only **reads** `job-pipeline/data/raw/*_raw.csv` and `data/sponsorship_companies.csv` and
never writes there or runs pipeline phases. It rebuilds from scratch on every run.
- It publishes **employer-owned job boards only** (Workday, Greenhouse, Lever, iCIMS, career pages …).
  Indeed/LinkedIn/Glassdoor/Zip, regulatoryjobs.com, association career centers and sources switched
  off for legitimacy are dropped.
- It keeps RA / QA / QC / Document Control titles that show a regulated-industry signal (FDA, GMP,
  ISO 13485, pharma …), drops software QA, and drops jobs posted more than 45 days ago (`--max-age`).
- H-1B counts come from public DOL LCA data. No company directory is published, and association
  member lists (AdvaMed, PBOA) never appear on the site.

## Files
- `site/vocab.json`: RA/QA terms, job families and levels, shared by `export.py` and the browser.
- `site/match.js`: scoring (55 skills, weighted by rarity · 20 family fit · 15 experience · 10 industry)
  and tips. No DOM, so it also runs in Node.
- Blend: the keyword score is combined with two similarity signals, as per-résumé z-scores weighted
  0.3 keyword / 0.3 TF-IDF / 0.4 embeddings, then mapped back to the keyword score's scale.
  `tools/embed_jobs.mjs` (run by `export.py`; needs `npm install`) builds both job-side indexes:
  - `site/lexical.json` + `lexical.bin`: TF-IDF (unigrams + bigrams, same as scikit-learn). About 5 MB,
    loaded with the page, so résumés get keyword + TF-IDF (0.3 / 0.7) at once.
  - `site/vectors.bin`: gte-base embeddings of each job's ~180-word chunks. The browser embeds the résumé with
    the same model (transformers.js, about 110 MB, downloaded once) and uses the best chunk similarity.
  If the model can't load, the site keeps keyword + TF-IDF.
- Evaluation: 40 résumés (24 real public ones, 16 synthetic), 1,860 LLM-labeled pairs, 21 embedding models,
  5 cross-encoders and a LightGBM ranker were compared (`tools/eval/`, results in `tools/results/general_eval.md`).
  `tools/eval/site_scores.mjs` + `eval_site.py` score the site's own code (`tools/results/site_eval.md`).
- `site/app.js`, `index.html`, `style.css`: the UI. PDF and DOCX are read with pdf.js and mammoth from cdnjs.
  Saved / applied / hidden jobs and "new since your last visit" live in the visitor's own browser
  (localStorage, keys `rqa.*`); there are no accounts and nothing is sent anywhere.

## Not in v1
Automatic daily refresh (the system is built, but nothing triggers it), LLM
explanations for top matches (needs a server and API key), cleaner company names (Workday tenants
like "Thermofisher", "Bdx"), dead-link checks, and outcome logging.
