# RA/QA Job Match

A public job board for Regulatory Affairs / Quality roles. Visitors drop in a résumé and get strong and
weak matches plus tips for widening their search. The résumé is parsed and matched **in the browser**
and never uploaded.

## Run locally
```
py export.py                                   # rebuild site/jobs.json from the pipeline
py -m http.server 8765 --directory site        # open http://localhost:8765
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
- `site/app.js`, `index.html`, `style.css`: the UI. PDF and DOCX are read with pdf.js and mammoth from cdnjs.

## Not in v1
Daily auto-refresh after the 5 AM run, public hosting, browser-side embedding similarity, LLM
explanations for top matches (needs a server and API key), cleaner company names (Workday tenants
like "Thermofisher", "Bdx"), dead-link checks, and outcome logging.
