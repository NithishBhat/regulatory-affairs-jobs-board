// Compare résumé→job ranking methods against labeled scores.
//   node tools/eval_rank.mjs <eval.json> [model ...]
// eval.json: {resume, jobs: [{id, title, description, terms, family, seniority, years, label}]}
// Labels are rubric scores (0–100) from an LLM that read each full description against the résumé.
import fs from "node:fs";
import { createRequire } from "node:module";
import { pipeline } from "@huggingface/transformers";

const require = createRequire(import.meta.url);
const Match = require("../site/match.js");
const vocab = JSON.parse(fs.readFileSync(new URL("../site/vocab.json", import.meta.url)));
const data = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const MODELS = process.argv.slice(3).length ? process.argv.slice(3)
  : ["Xenova/bge-small-en-v1.5", "Xenova/all-MiniLM-L6-v2", "Xenova/gte-small"];
const POOL = { "Xenova/bge-small-en-v1.5": "cls" };

// ---------- metrics ----------
const rank = (a) => { const idx = a.map((v, i) => [v, i]).sort((x, y) => x[0] - y[0]); const r = Array(a.length);
  for (let i = 0; i < idx.length;) { let j = i; while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
    for (let k = i; k <= j; k++) r[idx[k][1]] = (i + j) / 2; i = j + 1; } return r; };
const pearson = (a, b) => { const n = a.length, ma = a.reduce((s, x) => s + x, 0) / n, mb = b.reduce((s, x) => s + x, 0) / n;
  let num = 0, da = 0, db = 0; for (let i = 0; i < n; i++) { num += (a[i] - ma) * (b[i] - mb); da += (a[i] - ma) ** 2; db += (b[i] - mb) ** 2; }
  return num / Math.sqrt(da * db); };
const spearman = (a, b) => pearson(rank(a), rank(b));
const gain = (l) => (l >= 85 ? 3 : l >= 70 ? 2 : l >= 50 ? 1 : 0);
function ndcg(pred, labels, k = 20) {
  const dcg = (order) => order.slice(0, k).reduce((s, i, r) => s + (2 ** gain(labels[i]) - 1) / Math.log2(r + 2), 0);
  const byPred = pred.map((_, i) => i).sort((a, b) => pred[b] - pred[a]);
  const ideal = labels.map((_, i) => i).sort((a, b) => labels[b] - labels[a]);
  return dcg(byPred) / dcg(ideal);
}
const precAt = (pred, labels, k = 20) => pred.map((_, i) => i).sort((a, b) => pred[b] - pred[a]).slice(0, k)
  .filter((i) => labels[i] >= 70).length / k;
const z = (a) => { const m = a.reduce((s, x) => s + x, 0) / a.length; const sd = Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / a.length) || 1; return a.map((x) => (x - m) / sd); };
const report = (name, pred, labels) => console.log(
  `${name.padEnd(44)} spearman ${spearman(pred, labels).toFixed(3)}  nDCG@20 ${ndcg(pred, labels).toFixed(3)}  P@20(≥70) ${precAt(pred, labels).toFixed(2)}`);

// ---------- methods ----------
const jobs = data.jobs, labels = jobs.map((j) => j.label);
console.log(`${jobs.length} jobs; labels ≥70: ${labels.filter((l) => l >= 70).length}, ≥85: ${labels.filter((l) => l >= 85).length}`);
const ctx = Match.compile(vocab, jobs);
const profile = Match.analyzeResume(ctx, data.resume);
const kw = jobs.map((j) => Match.scoreJob(ctx, profile, j).score);
report("keyword (current match.js)", kw, labels);

const chunks = (text, words = 180) => { const w = text.split(/\s+/).filter(Boolean); const out = [];
  for (let i = 0; i < w.length; i += words) out.push(w.slice(i, i + words).join(" ")); return out.length ? out : [""]; };
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const meanVec = (vs) => { const m = new Float32Array(vs[0].length); for (const v of vs) for (let i = 0; i < m.length; i++) m[i] += v[i];
  const n = Math.hypot(...m); return m.map((x) => x / n); };

const results = {};
for (const model of MODELS) {
  const t0 = Date.now();
  const embed = await pipeline("feature-extraction", model, { dtype: "q8" });
  const opts = { pooling: POOL[model] || "mean", normalize: true };
  const vec = async (texts) => (await embed(texts, opts)).tolist();
  const resumeChunks = await vec(chunks(data.resume));
  const resumeMean = meanVec(resumeChunks);
  const head = [], mean = [], maxsim = [];
  for (const j of jobs) {
    const jc = await vec(chunks(`${j.title}. ${j.description}`));
    head.push(dot(resumeMean, jc[0]));                         // title + first ~180 words
    mean.push(dot(resumeMean, meanVec(jc)));                   // whole posting, averaged
    maxsim.push(Math.max(...jc.map((c) => Math.max(...resumeChunks.map((r) => dot(r, c))))));
  }
  console.log(`\n${model} (${((Date.now() - t0) / 1000).toFixed(0)}s)`);
  report("  emb: résumé-mean · job-head", head, labels);
  report("  emb: résumé-mean · job-mean", mean, labels);
  report("  emb: max chunk similarity", maxsim, labels);
  // Same, with job vectors stored as int16 (what the site ships).
  const q8 = (v) => { const s = Math.max(...v.map(Math.abs)) / 32767; return v.map((x) => Math.round(x / s) * s); };
  const maxsimQ = [];
  for (const j of jobs) {
    const jc = (await vec(chunks(`${j.title}. ${j.description}`))).map(q8);
    maxsimQ.push(Math.max(...jc.map((c) => Math.max(...resumeChunks.map((r) => dot(r, c))))));
  }
  report("  emb: max chunk similarity, int16 jobs", maxsimQ, labels);
  results[model] = { head, mean, maxsim, maxsimQ };
}

// ---------- is the blend really better? paired bootstrap vs keyword ----------
{
  const r = results[MODELS[0]], ze = z(r.maxsimQ), zk = z(kw);
  const blend = jobs.map((_, i) => 0.5 * ze[i] + 0.5 * zk[i]);
  report(`\n${MODELS[0].split("/")[1]} maxsim(int16) + kw, w=0.5`, blend, labels);
  let seed = 7; const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31);
  const dS = [], dN = [];
  for (let b = 0; b < 1000; b++) {
    const s = jobs.map(() => Math.floor(rnd() * jobs.length));
    const L = s.map((i) => labels[i]), B = s.map((i) => blend[i]), K = s.map((i) => kw[i]);
    dS.push(spearman(B, L) - spearman(K, L)); dN.push(ndcg(B, L) - ndcg(K, L));
  }
  const ci = (a) => { a.sort((x, y) => x - y); return `[${a[25].toFixed(3)}, ${a[974].toFixed(3)}], blend worse in ${a.filter((x) => x <= 0).length}/1000`; };
  console.log(`Δspearman 95% CI ${ci(dS)}\nΔnDCG@20  95% CI ${ci(dN)}`);
}

// ---------- blends: pick weight on one half, report on the other (2-fold) ----------
console.log("\nBlends (weight chosen on the other half; z-scored, w·emb + (1−w)·keyword):");
const idx = jobs.map((_, i) => i), folds = [idx.filter((i) => i % 2 === 0), idx.filter((i) => i % 2 === 1)];
for (const [model, r] of Object.entries(results)) for (const kind of ["mean", "maxsim"]) {
  const ze = z(r[kind]), zk = z(kw), outPred = Array(jobs.length); const ws = [];
  for (const [train, test] of [[folds[0], folds[1]], [folds[1], folds[0]]]) {
    let best = [-2, 0];
    for (let w = 0; w <= 1.0001; w += 0.1) {
      const p = train.map((i) => w * ze[i] + (1 - w) * zk[i]);
      const s = spearman(p, train.map((i) => labels[i])); if (s > best[0]) best = [s, w];
    }
    ws.push(best[1].toFixed(1));
    for (const i of test) outPred[i] = best[1] * ze[i] + (1 - best[1]) * zk[i];
  }
  report(`  ${model.split("/")[1]} ${kind} + kw (w=${ws.join("/")})`, outPred, labels);
  const fixed = jobs.map((_, i) => 0.7 * ze[i] + 0.3 * zk[i]);
  report(`  ${model.split("/")[1]} ${kind} + kw (w=0.7 fixed)`, fixed, labels);
}
