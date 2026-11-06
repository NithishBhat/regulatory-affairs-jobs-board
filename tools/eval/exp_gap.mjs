// Experience gap for every eval résumé × job, as the site sees it (site/match.js).
//   node tools/eval/exp_gap.mjs   -> cache/general_eval/scores/site_expgap.json {resume_id: {job_id: [gap, stated, level, résumé years known]}}
// gap = years the job asks for (stated, else typical for its title level) minus the résumé's years.
import fs from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const Match = require("../../site/match.js");
const root = new URL("../../", import.meta.url);
const json = (p) => JSON.parse(fs.readFileSync(new URL(p, root), "utf8"));
const jobs = json("site/jobs.json").jobs, resumes = json("cache/general_eval/resumes.json");
const ctx = Match.compile(json("site/vocab.json"), jobs);
const out = {};
for (const r of resumes) {
  const { years, yearsKnown } = Match.analyzeResume(ctx, r.text);
  out[r.id] = Object.fromEntries(jobs.map((j) => {
    const need = j.years != null ? j.years : Match.LEVEL_YEARS[j.seniority] || 0;
    return [j.id, [+(need - years).toFixed(1), j.years != null ? 1 : 0, j.seniority, yearsKnown ? 1 : 0]];
  }));
}
fs.writeFileSync(new URL("cache/general_eval/scores/site_expgap.json", root), JSON.stringify(out));
console.log(`${resumes.length} résumés × ${jobs.length} jobs`);
