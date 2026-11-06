// Tests for site/match.js and the built site files.   node --test tools/test/
import assert from "node:assert/strict";
import fs from "node:fs";
import { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const Match = require("../../site/match.js");
const root = new URL("../../", import.meta.url);
const json = (p) => JSON.parse(fs.readFileSync(new URL(p, root), "utf8"));
const buf = (p) => { const b = fs.readFileSync(new URL(p, root)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const vocab = json("site/vocab.json");
const data = json("site/jobs.json");
const ctx = Match.compile(vocab, data.jobs);
const NOW = new Date("2026-11-05");

const RA = `Regulatory Affairs Specialist
Acme Medical, Boston MA   Jan 2022 - Present
Prepared 510(k) submissions, EU MDR technical files, labeling review, ISO 13485, ISO 14971 risk management, eCTD.
Beta Devices   Jun 2020 - Dec 2021
Regulatory Affairs Associate: FDA 21 CFR 820 design controls, CAPA.
Education
Northeastern University, M.S. Regulatory Affairs   2018 - 2020`;

test("experience years: counts jobs, skips degrees, caps at today", () => {
  assert.equal(Match.experienceYears(RA, NOW), 6.5);   // Jan 2022–now (4.8) + Jun 2020–Dec 2021 (1.6); degree skipped
  assert.equal(Match.experienceYears("QA Analyst  2024 - 2030", NOW), 3);           // future end capped at Nov 2026
  assert.equal(Match.experienceYears("Bachelor of Science 2016 - 2020", NOW), 0);   // degree only
  assert.equal(Match.experienceYears("7+ years of quality experience. 2023 - Present", NOW), 7); // stated wins
});

test("résumé analysis: family, skills", () => {
  const p = Match.analyzeResume(ctx, RA, NOW);
  assert.equal(Object.entries(p.affinity).sort((a, b) => b[1] - a[1])[0][0], "RA");
  for (const id of ["510k", "mdr", "iso13485", "iso14971"]) assert.ok(!ctx.byId[id] || p.terms.has(id), id);
  assert.ok(p.terms.size >= 6);
});

test("scores: 0–100, RA résumé ranks RA jobs above QC jobs on average", () => {
  const p = Match.analyzeResume(ctx, RA, NOW);
  const scores = Match.scoreAll(ctx, p);
  assert.equal(scores.size, data.jobs.length);
  const mean = (fam) => { const v = data.jobs.filter((j) => j.family === fam).map((j) => scores.get(j.id).score);
    return v.reduce((a, b) => a + b, 0) / v.length; };
  for (const s of scores.values()) assert.ok(s.score >= 0 && s.score <= 100);
  assert.ok(mean("RA") > mean("QC") + 10, `RA ${mean("RA")} vs QC ${mean("QC")}`);
});

test("TF-IDF: built index matches what the site loads; self-similarity is highest", () => {
  const texts = json("cache/job_texts.json");
  const stop = json("tools/stop_words.json");
  const docs = texts.slice(0, 50);
  const built = Match.buildLexical(docs, stop);
  const lex = Match.loadLexical(built.meta, Buffer.concat([Buffer.from(built.post.buffer), Buffer.from(built.wt.buffer)]).buffer);
  const sims = Match.lexicalSims(lex, docs[7].text);
  const best = [...sims].sort((a, b) => b[1] - a[1])[0];
  assert.equal(best[0], docs[7].id);
  assert.ok(best[1] > 0.99);
  assert.equal(Match.lexicalSims(lex, "zzqx unrelated words").get(docs[0].id), 0);
});

test("site files are consistent with jobs.json", () => {
  const ids = new Set(data.jobs.map((j) => j.id));
  const lexMeta = json("site/lexical.json");
  assert.deepEqual(new Set(lexMeta.jobs), ids, "lexical.json jobs");
  const n = lexMeta.off[lexMeta.off.length - 1];
  assert.equal(fs.statSync(new URL("site/lexical.bin", root)).size, n * 4, "lexical.bin size");
  const vec = json("site/vectors.json");
  assert.equal(vec.model, Match.EMBED_MODEL);
  assert.deepEqual(new Set(Object.keys(vec.index)), ids, "vectors.json jobs");
  const rows = Object.values(vec.index).reduce((a, [, c]) => a + c, 0);
  assert.equal(fs.statSync(new URL("site/vectors.bin", root)).size, rows * vec.dim * 2, "vectors.bin size");
  assert.equal(data.count, data.jobs.length);
});

test("no non-US jobs in jobs.json", () => {
  const bad = data.jobs.filter((j) => /\((?:[mwfdhx]\s*\/\s*){1,3}[mwfdhx]\)/i.test(j.title) ||
    /,[^,]*,\s*(?!US\b)[A-Z]{2}\s*$/.test(j.location) || /riyadh|belgio|germany|india\b|canada/i.test(j.location));
  assert.deepEqual(bad.map((j) => j.location), []);
});

test("work-authorization limits: every job has a list; restricted jobs never count as sponsor-track", () => {
  const kinds = new Set(["citizen", "clearance", "export", "no_sponsor"]);
  for (const j of data.jobs) {
    assert.ok(Array.isArray(j.restrict), j.id);
    for (const k of j.restrict) assert.ok(kinds.has(k), k);
    if (j.restrict.length) assert.equal(Match.hasSponsorTrack(j), false, j.id);
  }
  assert.ok(data.jobs.some((j) => j.restrict.length), "some postings state limits");
});

test("blend: keeps the keyword scale, follows the signals", () => {
  const p = Match.analyzeResume(ctx, RA, NOW);
  const scores = Match.scoreAll(ctx, p);
  const kw = [...scores.values()].map((s) => s.score);
  const lex = Match.loadLexical(json("site/lexical.json"), buf("site/lexical.bin"));
  Match.blendSignals(scores, { lexical: Match.lexicalSims(lex, RA) });
  const out = [...scores.values()];
  const mean = (v) => v.reduce((a, b) => a + b, 0) / v.length;
  assert.ok(Math.abs(mean(out.map((s) => s.score)) - mean(kw)) < 2, "mean preserved");
  assert.ok(out.every((s) => s.keyword != null && s.score >= 0 && s.score <= 100));
  // A huge semantic signal for one job moves it to the top.
  const target = data.jobs[data.jobs.length - 1].id;
  const sem = new Map(data.jobs.map((j) => [j.id, j.id === target ? 10 : 0]));
  Match.blendSignals(scores, { lexical: Match.lexicalSims(lex, RA), semantic: sem });
  const top = Math.max(...[...scores.values()].map((s) => s.score));   // scores cap at 100, so ties are possible
  assert.equal(scores.get(target).score, top);
});

test("calibration: monotone, within 0–100, and loaded by the site", () => {
  const tables = json("site/calibration.json");
  assert.deepEqual(Object.keys(tables).sort(), ["full", "lexical"]);
  for (const cal of Object.values(tables)) {
    assert.equal(cal.x.length, cal.y.length);
    for (let i = 1; i < cal.y.length; i++) { assert.ok(cal.x[i] > cal.x[i - 1]); assert.ok(cal.y[i] >= cal.y[i - 1]); }
    let prev = -1;
    for (let raw = -20; raw <= 150; raw += 0.5) {
      const v = Match.calibrate(cal, raw);
      assert.ok(v >= 0 && v <= 100 && v >= prev - 1e-9, `raw ${raw}`);
      prev = v;
    }
    assert.ok(Match.calibrate(cal, 200) < 100, "a perfect 100 isn't reachable");
  }
});

test("experience: penalty only when the résumé is short of stated years, and only when years are known", () => {
  const job = { id: "x", terms: [], family: "QA", seniority: "senior", years: 5 };
  const c = Match.compile(vocab, [job]);
  const at = (years, yearsKnown) => {
    const scores = new Map([["x", { score: 80, raw: 80 }]]);
    Match.experienceAdjust(c, { years, yearsKnown }, scores);
    return scores.get("x").score;
  };
  assert.equal(at(4, true), 70);       // 1 year short: −10
  assert.equal(at(1, true), 70);       // capped at −10
  assert.equal(at(6, true), 80);       // meets it
  assert.equal(at(0, false), 80);      // years unknown: no penalty
});

test("finalize: rank orders jobs, score is the calibrated number", () => {
  const p = Match.analyzeResume({ ...ctx }, RA, NOW);
  const cctx = Match.compile(vocab, data.jobs, json("site/calibration.json"));
  const scores = Match.scoreAll(cctx, p);
  const lex = Match.loadLexical(json("site/lexical.json"), buf("site/lexical.bin"));
  Match.finalize(cctx, p, scores, { lexical: Match.lexicalSims(lex, RA) });
  const v = [...scores.values()];
  assert.ok(v.every((s) => Number.isFinite(s.rank) && Number.isInteger(s.score) && s.score >= 0 && s.score < 100));
  const byRank = [...v].sort((a, b) => b.rank - a.rank);
  for (let i = 1; i < byRank.length; i++) assert.ok(byRank[i].score <= byRank[i - 1].score, "shown score follows the ranking");
});

test("chunks and maxSim", () => {
  const words = Array.from({ length: 400 }, (_, i) => `w${i}`).join(" ");
  assert.equal(Match.chunks(words).length, 3);
  assert.deepEqual(Match.chunks(""), [""]);
  const rows = new Int16Array([32767, 0, 0, 32767]);                 // two unit vectors, dim 2
  assert.ok(Math.abs(Match.maxSim([[1, 0]], rows, 0, 2, 2) - 1) < 1e-9);
  assert.ok(Math.abs(Match.maxSim([[0, 1]], rows, 0, 1, 2)) < 1e-9);
});

test("tips run without errors", () => {
  const p = Match.analyzeResume(ctx, RA, NOW);
  const scores = Match.scoreAll(ctx, p);
  const labels = Object.fromEntries(vocab.families.map((f) => [f.id, f.label]));
  const t = Match.tips(ctx, p, scores, data.jobs, data.jobs, { families: new Set(["RA", "QA", "QC", "DOC"]), age: 45 }, labels);
  assert.ok(Array.isArray(t));
  for (const x of t) assert.ok(x.title && x.body);
});
