// Keyword-layer scores (site/match.js) for every résumé × job, used for pooling.
//   node tools/eval/keyword_scores.mjs cache/general_eval/resumes.json cache/general_eval/keyword_scores.json
import fs from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const Match = require("../../site/match.js");
const root = new URL("../../", import.meta.url);
const vocab = JSON.parse(fs.readFileSync(new URL("site/vocab.json", root)));
const jobs = JSON.parse(fs.readFileSync(new URL("site/jobs.json", root))).jobs;
const resumes = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const ctx = Match.compile(vocab, jobs);
const out = {};
for (const r of resumes) {
  const profile = Match.analyzeResume(ctx, r.text);
  out[r.id] = Object.fromEntries(jobs.map((j) => [j.id, Match.scoreJob(ctx, profile, j).score]));
}
fs.writeFileSync(process.argv[3], JSON.stringify(out));
console.log(`${resumes.length} résumés × ${jobs.length} jobs`);
