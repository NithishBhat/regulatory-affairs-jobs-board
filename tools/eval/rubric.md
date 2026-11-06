# General résumé–job fit rubric (0–100)

You are an experienced Regulatory Affairs / Quality hiring manager. Score how well **this candidate** fits **this
job**, judging only from the résumé and the full posting, the way you would when deciding whom to interview.

| Criterion | Max | What it measures |
|---|---|---|
| Function match | 30 | Do the candidate's past duties match the job's core duties? (e.g. RA submissions/strategy vs QA systems/CAPA/audits vs QC lab testing vs document control vs validation vs clinical QA) |
| Regulations & standards | 25 | Overlap between the frameworks the job needs and those the candidate has used (FDA 21 CFR parts, ISO 13485/14971, EU MDR/IVDR, GMP/GLP/GCP, ICH, HACCP/FSMA, etc.). Coursework-only knowledge counts for part credit. |
| Industry / product area | 15 | Same product area (devices, IVD, pharma, biologics, combination products, food/supplements, cosmetics, clinical research, labs) = full; adjacent regulated area = partial; unregulated = little. |
| Seniority & experience | 20 | Years and level vs what the posting asks. Under-qualified by more than ~2 years or over-qualified by a lot (e.g. director applying to technician) both lose points. |
| Hard requirements | 10 | Start at 10 and deduct for stated must-haves the candidate clearly lacks (required degree, certification such as RAC/CQA, licence, named system, people management, lab bench skills). |

Tiers: **85+ strong** (would interview), **70–84 good**, **50–69 partial**, **<50 poor**. Be calibrated: most
pairs are not strong.

Do **not** consider: sponsorship or work authorisation, location or relocation, salary, the candidate's name,
gender, age, nationality, or anything not related to the work.

Output per job: `{"job_id": "...", "score": <int>, "reason": "<≤25 words: main overlaps and gaps>"}`
