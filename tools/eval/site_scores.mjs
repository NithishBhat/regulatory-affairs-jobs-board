// Score every eval résumé × job with the site's own code (site/match.js + the files in site/), so the
// evaluation measures exactly what visitors get.
//   node tools/eval/site_scores.mjs [--no-semantic] [--reuse-semantic]
// Writes cache/general_eval/scores/site_{keyword,lexical,semantic,final,shown}.json {resume_id: {job_id: value}}:
// final = the ranking value (s.rank), shown = the number on the card (calibrated s.score).
// --reuse-semantic reads site_semantic.json instead of re-embedding the résumés.
import fs from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const Match = require("../../site/match.js");
const root = new URL("../../", import.meta.url);
const read = (p) => fs.readFileSync(new URL(p, root));
const json = (p) => JSON.parse(read(p));
const out = (name, data) => fs.writeFileSync(new URL(`cache/general_eval/scores/site_${name}.json`, root), JSON.stringify(data));

const vocab = json("site/vocab.json"), jobs = json("site/jobs.json").jobs;
const resumes = json("cache/general_eval/resumes.json");
const calPath = new URL("site/calibration.json", root);
const ctx = Match.compile(vocab, jobs, fs.existsSync(calPath) ? JSON.parse(fs.readFileSync(calPath)) : null);
const buf = (p) => { const b = read(p); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const lex = Match.loadLexical(json("site/lexical.json"), buf("site/lexical.bin"));

// What a visitor gets (Match.finalize): s.rank orders the jobs, s.score is the shown (calibrated) number.
// Ranking metrics and the calibration fit use rank, which doesn't depend on the calibration.
const shown = {};
const final = (r, signals) => {
  const profile = Match.analyzeResume(ctx, r.text);
  const scores = Match.scoreAll(ctx, profile);
  Match.finalize(ctx, profile, scores, signals);
  shown[r.id] = Object.fromEntries([...scores].map(([id, s]) => [id, s.score]));
  return Object.fromEntries([...scores].map(([id, s]) => [id, s.rank]));
};

const kw = {}, lx = {}, fin = {};
for (const r of resumes) {
  const profile = Match.analyzeResume(ctx, r.text);
  kw[r.id] = Object.fromEntries(jobs.map((j) => [j.id, Match.scoreJob(ctx, profile, j).score]));
  const sims = Match.lexicalSims(lex, r.text);
  lx[r.id] = Object.fromEntries(sims);
  fin[r.id] = final(r, { lexical: sims });
}
out("keyword", kw);
out("lexical", lx);
out("final_no_semantic", fin);
console.log(`${resumes.length} résumés × ${jobs.length} jobs: keyword, lexical`);

if (process.argv.includes("--reuse-semantic")) {
  const sem = json("cache/general_eval/scores/site_semantic.json"), all = {};
  for (const r of resumes) all[r.id] = final(r, { lexical: Match.lexicalSims(lex, r.text), semantic: new Map(Object.entries(sem[r.id])) });
  out("final", all);
  out("shown", shown);
  console.log("semantic (reused)");
} else if (!process.argv.includes("--no-semantic")) {
  const { pipeline } = await import("@huggingface/transformers");
  const meta = json("site/vectors.json"), rows = new Int16Array(buf("site/vectors.bin"));
  const embed = await pipeline("feature-extraction", meta.model, { dtype: "q8" });
  const sem = {}, all = {};
  for (const r of resumes) {
    const vecs = (await embed(Match.chunks(r.text), { pooling: "mean", normalize: true })).tolist();
    const sims = new Map(Object.entries(meta.index).map(([id, [first, n]]) =>
      [id, Match.maxSim(vecs, rows, first, n, meta.dim)]));
    sem[r.id] = Object.fromEntries(sims);
    all[r.id] = final(r, { lexical: Match.lexicalSims(lex, r.text), semantic: sims });
  }
  out("semantic", sem);
  out("final", all);
  out("shown", shown);
  console.log(`semantic (${meta.model})`);
}
