// Try popular résumé→job ranking strategies (Medium posts / GitHub repos) against labeled jobs.
//   node tools/strategies.mjs [cache/eval.json]
// Labels: rubric scores (0–100) from an LLM that read each full posting against the résumé.
// Per-strategy scores are cached in cache/strategy_scores/ (re-runs are cheap); the summary goes to
// tools/results/strategies.{md,json}. Blend weights are chosen on one half of the jobs and scored on the other.
import fs from "node:fs";
import { createRequire } from "node:module";
import { AutoModelForSequenceClassification, AutoTokenizer, pipeline } from "@huggingface/transformers";

const require = createRequire(import.meta.url);
const Match = require("../site/match.js");
const ROOT = new URL("../", import.meta.url);
const vocab = JSON.parse(fs.readFileSync(new URL("site/vocab.json", ROOT)));
const data = JSON.parse(fs.readFileSync(process.argv[2] || new URL("cache/eval.json", ROOT), "utf8"));
const jobs = data.jobs, labels = jobs.map((j) => j.label), N = jobs.length;
const SCORES = new URL("cache/strategy_scores/", ROOT);
fs.mkdirSync(SCORES, { recursive: true });
fs.mkdirSync(new URL("tools/results/", ROOT), { recursive: true });

// ---------- metrics ----------
const rankOf = (a) => { const idx = a.map((v, i) => [v, i]).sort((x, y) => x[0] - y[0]); const r = Array(a.length);
  for (let i = 0; i < idx.length;) { let j = i; while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
    for (let k = i; k <= j; k++) r[idx[k][1]] = (i + j) / 2; i = j + 1; } return r; };
const pearson = (a, b) => { const n = a.length, ma = a.reduce((s, x) => s + x, 0) / n, mb = b.reduce((s, x) => s + x, 0) / n;
  let num = 0, da = 0, db = 0; for (let i = 0; i < n; i++) { num += (a[i] - ma) * (b[i] - mb); da += (a[i] - ma) ** 2; db += (b[i] - mb) ** 2; }
  return num / Math.sqrt(da * db); };
const spearman = (a, b) => pearson(rankOf(a), rankOf(b));
const gain = (l) => (l >= 85 ? 3 : l >= 70 ? 2 : l >= 50 ? 1 : 0);
const order = (p) => p.map((_, i) => i).sort((a, b) => p[b] - p[a]);
function ndcg(p, L, k = 20) {
  const dcg = (o) => o.slice(0, k).reduce((s, i, r) => s + (2 ** gain(L[i]) - 1) / Math.log2(r + 2), 0);
  return dcg(order(p)) / dcg(order(L));
}
const precAt = (p, L, k = 20) => order(p).slice(0, k).filter((i) => L[i] >= 70).length / k;
const z = (a) => { const m = a.reduce((s, x) => s + x, 0) / a.length;
  const sd = Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / a.length) || 1; return a.map((x) => (x - m) / sd); };
const metrics = (p) => ({ spearman: spearman(p, labels), ndcg20: ndcg(p, labels), p20: precAt(p, labels) });

// ---------- text helpers ----------
const STOP = new Set(("a an the and or of to in for on with at by from as is are be this that will you your our we " +
  "us it its their they or not all any can may must have has including such other job role work team ability " +
  "experience years year strong skills required preferred").split(" "));
const tokens = (t) => String(t).toLowerCase().match(/[a-z0-9][a-z0-9+#().\/-]*[a-z0-9)]|[a-z0-9]/g)?.filter((w) => !STOP.has(w)) || [];
const jobText = (j) => `${j.title}. ${j.description}`;
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const unit = (v) => { const n = Math.hypot(...v) || 1; return v.map((x) => x / n); };
const meanVec = (vs) => unit(vs[0].map((_, i) => vs.reduce((s, v) => s + v[i], 0)));

// "Requirements" part of a posting (JobMatch-AI-style requirement focus, without an LLM): from the first
// qualifications / responsibilities heading, up to boilerplate (EEO, benefits, about us).
const REQ_HEAD = /(basic|minimum|required|preferred)?\s*(qualifications|requirements|what you('|’)ll (need|bring|do)|who you are|what we('|’)re looking for|responsibilities|duties|essential functions|your role)/i;
const TAIL = /(equal (employment )?opportunity|\beeo\b|benefits|about (us|the company)|we offer|compensation|pay range|salary range)/i;
function requirements(desc) {
  const m = desc.search(REQ_HEAD);
  let s = m >= 0 ? desc.slice(m) : desc;
  const t = s.slice(200).search(TAIL);
  if (t >= 0) s = s.slice(0, 200 + t);
  return s.split(/\s+/).length > 40 ? s : desc;
}

// ---------- strategies (each returns one score per job) ----------
function tfidf(ngram = 2) {
  const grams = (t) => { const w = tokens(t), out = [...w]; if (ngram > 1) for (let i = 0; i + 1 < w.length; i++) out.push(w[i] + " " + w[i + 1]); return out; };
  const docs = jobs.map((j) => grams(jobText(j))), q = grams(data.resume), df = new Map();
  for (const d of docs) for (const g of new Set(d)) df.set(g, (df.get(g) || 0) + 1);
  const vec = (d) => { const tf = new Map(); for (const g of d) tf.set(g, (tf.get(g) || 0) + 1);
    const v = new Map(); let n = 0;
    for (const [g, c] of tf) { const w = (1 + Math.log(c)) * (Math.log((N + 1) / ((df.get(g) || 0) + 1)) + 1); v.set(g, w); n += w * w; }
    n = Math.sqrt(n) || 1; for (const [g, w] of v) v.set(g, w / n); return v; };
  const qv = vec(q);
  return docs.map((d) => { const v = vec(d); let s = 0; for (const [g, w] of qv) if (v.has(g)) s += w * v.get(g); return s; });
}

function bm25(k1 = 1.5, b = 0.75) {
  const docs = jobs.map((j) => tokens(jobText(j))), avg = docs.reduce((s, d) => s + d.length, 0) / N, df = new Map();
  for (const d of docs) for (const w of new Set(d)) df.set(w, (df.get(w) || 0) + 1);
  const q = [...new Set(tokens(data.resume))];
  return docs.map((d) => { const tf = new Map(); for (const w of d) tf.set(w, (tf.get(w) || 0) + 1);
    return q.reduce((s, w) => { const f = tf.get(w); if (!f) return s;
      const idf = Math.log(1 + (N - df.get(w) + 0.5) / (df.get(w) + 0.5));
      return s + idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * d.length / avg)); }, 0); });
}

const ctx = Match.compile(vocab, jobs), profile = Match.analyzeResume(ctx, data.resume);
const keyword = () => jobs.map((j) => Match.scoreJob(ctx, profile, j).score);
const skillSets = () => jobs.map((j) => j.terms.filter((t) => ctx.byId[t] && ctx.byId[t].cat !== "industry"));
const jaccard = () => skillSets().map((t) => { const inter = t.filter((x) => profile.terms.has(x)).length;
  const resumeSkills = [...profile.terms].filter((x) => ctx.byId[x].cat !== "industry").length;
  return inter / (t.length + resumeSkills - inter || 1); });
const coverage = () => skillSets().map((t) => t.length ? t.filter((x) => profile.terms.has(x)).length / t.length : 0);

const EMB = {
  "MiniLM-L6": { id: "Xenova/all-MiniLM-L6-v2" },
  "bge-small": { id: "Xenova/bge-small-en-v1.5", pool: "cls", q: "Represent this sentence for searching relevant passages: " },
  "gte-small": { id: "Xenova/gte-small" },
  "bge-base": { id: "Xenova/bge-base-en-v1.5", pool: "cls", q: "Represent this sentence for searching relevant passages: " },
  "e5-base": { id: "Xenova/e5-base-v2", q: "query: ", p: "passage: " },
  "gte-base": { id: "Xenova/gte-base" },
};
const embedders = {};
async function embedder(name) {
  const m = EMB[name];
  embedders[name] = embedders[name] || await pipeline("feature-extraction", m.id, { dtype: "q8" });
  return async (texts, kind) => (await embedders[name](texts.map((t) => (kind === "q" ? m.q || "" : m.p || "") + t),
    { pooling: m.pool || "mean", normalize: true })).tolist();
}
async function embedScores(name, { section = false } = {}) {
  const e = await embedder(name);
  const rc = await e(Match.chunks(data.resume), "q"), rm = meanVec(rc);
  const mean = [], maxsim = [];
  for (const j of jobs) {
    const jc = await e(Match.chunks(section ? `${j.title}. ${requirements(j.description)}` : jobText(j)), "p");
    mean.push(dot(rm, meanVec(jc)));
    maxsim.push(Math.max(...jc.map((c) => Math.max(...rc.map((r) => dot(r, c))))));
  }
  return { mean, maxsim };
}

async function crossScores(modelId, idxs) {
  const tok = await AutoTokenizer.from_pretrained(modelId);
  const model = await AutoModelForSequenceClassification.from_pretrained(modelId, { dtype: "q8" });
  const rc = Match.chunks(data.resume), out = Array(N).fill(null);
  for (const i of idxs) {
    const jc = Match.chunks(jobText(jobs[i]));
    const q = [], d = [];
    for (const r of rc) for (const c of jc) { q.push(r); d.push(c); }
    const { logits } = await model(tok(q, { text_pair: d, padding: true, truncation: true }));
    out[i] = Math.max(...logits.tolist().map((x) => x[0]));
  }
  return out;
}

// ---------- run with caching ----------
async function cached(name, fn) {
  const f = new URL(`${name}.json`, SCORES);
  if (fs.existsSync(f)) { const c = JSON.parse(fs.readFileSync(f)); if (c.ids.join() === jobs.map((j) => j.id).join()) return c.value; }
  const t0 = Date.now(), value = await fn();
  fs.writeFileSync(f, JSON.stringify({ ids: jobs.map((j) => j.id), value, secs: (Date.now() - t0) / 1000 }));
  console.log(`  computed ${name} in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  return value;
}

const rows = [];
const add = (name, source, p, note = "") => { const m = metrics(p); rows.push({ name, source, ...m, note });
  console.log(`${name.padEnd(46)} ρ ${m.spearman.toFixed(3)}  nDCG@20 ${m.ndcg20.toFixed(3)}  P@20 ${m.p20.toFixed(2)}`); };

// Out-of-fold weight for w·A + (1−w)·B on z-scores (2 folds, even/odd jobs).
function blendOOF(a, b) {
  const za = z(a), zb = z(b), out = Array(N), ws = [];
  const folds = [[0, 1], [1, 0]].map(([tr, te]) => [[...Array(N).keys()].filter((i) => i % 2 === tr), [...Array(N).keys()].filter((i) => i % 2 === te)]);
  for (const [train, test] of folds) {
    let best = [-2, 0];
    for (let w = 0; w <= 1.0001; w += 0.1) { const s = spearman(train.map((i) => w * za[i] + (1 - w) * zb[i]), train.map((i) => labels[i])); if (s > best[0]) best = [s, w]; }
    ws.push(+best[1].toFixed(1)); for (const i of test) out[i] = best[1] * za[i] + (1 - best[1]) * zb[i];
  }
  return [out, ws];
}
const rrf = (...lists) => { const r = lists.map((l) => { const o = order(l), rk = Array(N); o.forEach((i, k) => { rk[i] = k; }); return rk; });
  return [...Array(N).keys()].map((i) => r.reduce((s, rk) => s + 1 / (60 + rk[i]), 0)); };
// Rerank the top K of a first-stage list by a second score; the rest keep their first-stage order.
const rerankTop = (first, second, K = 60) => { const top = order(first).slice(0, K), zs = z(top.map((i) => second[i]));
  const out = first.slice(); top.forEach((i, k) => { out[i] = 1e6 + zs[k]; }); return out; };

console.log(`${N} labeled jobs (≥70: ${labels.filter((l) => l >= 70).length}, ≥85: ${labels.filter((l) => l >= 85).length})\n`);
const S = {};
S.keyword = await cached("keyword", async () => keyword());
S.tfidf1 = await cached("tfidf_unigram", async () => tfidf(1));
S.tfidf2 = await cached("tfidf_bigram", async () => tfidf(2));
S.bm25 = await cached("bm25", async () => bm25());
S.jaccard = await cached("skill_jaccard", async () => jaccard());
S.coverage = await cached("skill_coverage", async () => coverage());
for (const name of Object.keys(EMB)) S[name] = await cached(`emb_${name}`, () => embedScores(name));
S.gteReq = await cached("emb_gte-small_requirements", () => embedScores("gte-small", { section: true }));
S.cross = await cached("cross_msmarco_all", () => crossScores("Xenova/ms-marco-MiniLM-L-6-v2", [...Array(N).keys()]));
// Computed on the GPU by tools/gpu_scores.py (too slow here); skipped when absent.
const gpu = (name) => { const f = new URL(`${name}.json`, SCORES); if (!fs.existsSync(f)) return null;
  const c = JSON.parse(fs.readFileSync(f)); return c.ids.join() === jobs.map((j) => j.id).join() ? c.value : null; };
const G = { bgeRerank: gpu("cross_bge_reranker_all"), bgeV2m3: gpu("cross_bge_reranker_v2m3_all"),
  "mpnet-base": gpu("emb_mpnet-base"), "bge-large": gpu("emb_bge-large"), "e5-large": gpu("emb_e5-large"), "gte-large": gpu("emb_gte-large") };

add("Keyword vocab (current site, layer 1)", "this project", S.keyword);
add("TF-IDF cosine, unigrams", "classic Medium tutorials", S.tfidf1);
add("TF-IDF cosine, uni+bigrams", "classic Medium tutorials", S.tfidf2);
add("BM25 (résumé as query)", "rank_bm25 / HF resume-screener demo", S.bm25);
add("Skill Jaccard (vocab terms)", "skill-extractor repos (SkillTool etc.)", S.jaccard);
add("Skill coverage (job terms found)", "Resume-Matcher keyword %", S.coverage);
for (const name of Object.keys(EMB)) {
  add(`${name}: mean embedding cosine`, "SBERT-cosine repos", S[name].mean);
  add(`${name}: max chunk similarity`, "chunked retrieval", S[name].maxsim);
}
add("gte-small max-sim, requirements section only", "JobMatch-AI requirement focus", S.gteReq.maxsim);
add("Cross-encoder ms-marco-MiniLM (all jobs)", "HF resume-screener demo", S.cross);
if (G.bgeRerank) add("Cross-encoder bge-reranker-base (all jobs, GPU)", "reranker posts", G.bgeRerank);
if (G.bgeV2m3) add("Cross-encoder bge-reranker-v2-m3 (all jobs, GPU)", "reranker posts", G.bgeV2m3);
for (const name of ["mpnet-base", "bge-large", "e5-large", "gte-large"]) if (G[name]) {
  add(`${name}: mean embedding cosine (GPU)`, name === "mpnet-base" ? "most-used SBERT in Medium posts" : "large-model ceiling", G[name].mean);
  add(`${name}: max chunk similarity (GPU)`, "large-model ceiling", G[name].maxsim);
}

console.log("\nHybrids (weights out-of-fold):");
const hyb = (label, src, a, b) => { const [p, ws] = blendOOF(a, b); add(`${label} (w=${ws.join("/")})`, src, p); return p; };
hyb("TF-IDF + MiniLM mean", "lisan-5/resume-job-matching hybrid", S["MiniLM-L6"].mean, S.tfidf2);
hyb("BM25 + gte-small max-sim", "hybrid search posts", S["gte-small"].maxsim, S.bm25);
const prod = (() => { const a = z(S["gte-small"].maxsim), b = z(S.keyword); return a.map((x, i) => 0.5 * x + 0.5 * b[i]); })();
add("Keyword + gte-small max-sim, w=0.5 (current site)", "this project", prod);
for (const name of ["bge-small", "bge-base", "e5-base", "gte-base"]) hyb(`Keyword + ${name} max-sim`, "this project, bigger models", S[name].maxsim, S.keyword);
hyb("Keyword + gte-small requirements max-sim", "this project + requirement focus", S.gteReq.maxsim, S.keyword);
hyb("Keyword + cross-encoder", "hybrid + reranker", S.cross, S.keyword);
add("RRF(keyword, BM25, gte-small max-sim)", "reciprocal rank fusion posts (k=60)", rrf(S.keyword, S.bm25, S["gte-small"].maxsim));
add("RRF(keyword, gte-small max-sim, cross-encoder)", "RRF + reranker", rrf(S.keyword, S["gte-small"].maxsim, S.cross));
add("Current site → rerank top 60 by cross-encoder", "retrieve-then-rerank posts", rerankTop(prod, S.cross));
for (const [key, label] of [["bgeRerank", "bge-reranker-base"], ["bgeV2m3", "bge-reranker-v2-m3"]]) if (G[key]) {
  add(`Current site → rerank top 60 by ${label}`, "retrieve-then-rerank posts", rerankTop(prod, G[key]));
  hyb(`Keyword + ${label}`, "hybrid + reranker", G[key], S.keyword);
}
for (const name of ["bge-large", "e5-large", "gte-large"]) if (G[name]) hyb(`Keyword + ${name} max-sim (GPU)`, "large-model ceiling", G[name].maxsim, S.keyword);

// ---------- store ----------
rows.sort((a, b) => b.spearman - a.spearman);
const sizes = { "MiniLM-L6": 23, "bge-small": 34, "gte-small": 34, "bge-base": 110, "e5-base": 110, "gte-base": 110 };
const md = [`# Résumé→job ranking strategies — ${new Date().toISOString().slice(0, 10)}`, "",
  `${N} jobs scored by an LLM against one RA/QA résumé (rubric 0–100; ≥70 = good). ρ = Spearman with those`,
  "scores over all jobs; nDCG@20 / P@20 = quality of the top 20. Blend weights chosen on one half of the jobs,",
  "scored on the other. Single résumé + LLM labels: treat differences under ~0.03 ρ as noise.", "",
  "| # | Strategy | From | ρ | nDCG@20 | P@20 |", "|---|---|---|---|---|---|",
  ...rows.map((r, i) => `| ${i + 1} | ${r.name} | ${r.source} | ${r.spearman.toFixed(3)} | ${r.ndcg20.toFixed(3)} | ${r.p20.toFixed(2)} |`),
  "", "Approx. q8 model download for in-browser use (MB): " + Object.entries(sizes).map(([k, v]) => `${k} ${v}`).join(", ") +
  ", ms-marco-MiniLM cross-encoder 23, bge-reranker-base ~280, bge-reranker-v2-m3 ~570; *-large ~335 (GPU-only rows are not browser-practical).", "",
  "Not run: LLM-based extraction / judging (JobMatch-AI two-pass, ConFit v2 hypothetical résumé) — need an LLM",
  "per job; SPLADE (no transformers.js port); fine-tuned models (ConFit, conSultantBERT) — need labeled pairs at scale."];
fs.writeFileSync(new URL("tools/results/strategies.md", ROOT), md.join("\n") + "\n");
fs.writeFileSync(new URL("tools/results/strategies.json", ROOT), JSON.stringify({ date: new Date().toISOString(), n: N, rows }, null, 1));
console.log("\n-> tools/results/strategies.md, strategies.json");
