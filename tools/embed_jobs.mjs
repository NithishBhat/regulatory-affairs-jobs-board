// Precompute the job-side indexes for matching in the browser.
//   node tools/embed_jobs.mjs        (export.py runs this after writing site/jobs.json)
// Reads cache/job_texts.json [{id, text}] and writes:
//   site/lexical.json + lexical.bin   TF-IDF index (Match.buildLexical; stop words from tools/stop_words.json)
//   site/vectors.bin + vectors.json   embeddings (Int16, unit vectors × 32767, one row per chunk) and
//                                     {model, dim, index: {job_id: [firstRow, rows]}}
//   --lexical-only                    skip the embeddings
import fs from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const Match = require("../site/match.js");
const root = new URL("../", import.meta.url);
const texts = JSON.parse(fs.readFileSync(new URL("cache/job_texts.json", root), "utf8"));

const stop = JSON.parse(fs.readFileSync(new URL("tools/stop_words.json", root), "utf8"));
const lex = Match.buildLexical(texts, stop);
fs.writeFileSync(new URL("site/lexical.json", root), JSON.stringify(lex.meta));
fs.writeFileSync(new URL("site/lexical.bin", root), Buffer.concat([Buffer.from(lex.post.buffer), Buffer.from(lex.wt.buffer)]));
console.log(`${lex.meta.terms.length} TF-IDF terms, ${lex.post.length} postings -> site/lexical.bin`);
if (process.argv.includes("--lexical-only")) process.exit(0);

const { pipeline } = await import("@huggingface/transformers");
const embed = await pipeline("feature-extraction", Match.EMBED_MODEL, { dtype: "q8" });
const rows = [], index = {};
for (const { id, text } of texts) {
  const vecs = (await embed(Match.chunks(text), { pooling: "mean", normalize: true })).tolist();
  index[id] = [rows.length, vecs.length];
  rows.push(...vecs);
}
const dim = rows[0].length, out = new Int16Array(rows.length * dim);
rows.forEach((v, r) => v.forEach((x, i) => { out[r * dim + i] = Math.round(x * 32767); }));
fs.writeFileSync(new URL("site/vectors.bin", root), Buffer.from(out.buffer));
fs.writeFileSync(new URL("site/vectors.json", root), JSON.stringify({ model: Match.EMBED_MODEL, dim, index }));
console.log(`${texts.length} jobs, ${rows.length} chunks -> site/vectors.bin (${(out.byteLength / 1e6).toFixed(1)} MB)`);
