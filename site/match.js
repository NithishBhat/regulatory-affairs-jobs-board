// Résumé ↔ job matching. Pure functions (no DOM) so they run in the browser and in Node tests.
(function (root) {
  "use strict";

  // Tab cut-offs on the shown (calibrated) score, which estimates a hiring manager's 0–100 rating (70+ good fit).
  // Because it's an estimate, good jobs often show in the low 60s, so a cut at 70 hid most of them. 58 balances
  // precision and recall against the labels (tools/results/calibration.md): 78% of "Best matches" are at least
  // a partial fit and half of all good fits land there. "Worth a look" (44+): 61% at least partial, 81% recall.
  const STRONG = 58, WEAK = 44;
  const SMOOTH = 6; // idf units added to every job's total, so jobs with few listed skills can't score 100
  // Typical minimum years by title level, used when a posting states no number.
  const LEVEL_YEARS = { entry: 0, mid: 2, senior: 5, manager: 6, director: 10 };
  const FAMILY_CATS = { RA: ["submission", "standard"], QA: ["quality", "standard"], QC: ["lab"], DOC: [] };
  const DOC_TERMS = ["doccontrol", "sop", "gdp", "part11"];
  const SKIP_TIP = new Set(["fda", "audit", "sop"]); // too common to be useful advice
  const MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec";
  const RANGE = new RegExp(
    "(?:(" + MONTHS + ")[a-z]*\\.?\\s*)?((?:19|20)\\d{2})\\s*(?:-|–|—|to)\\s*" +
    "(?:(present|current|now|today)|(?:(" + MONTHS + ")[a-z]*\\.?\\s*)?((?:19|20)\\d{2}))", "gi");
  // A date range on a line like this is a degree, not a job. (No bare "MS"/"BS": they're also state codes.)
  const EDUCATION = /\b(?:university|college|school|bachelor|masters?\b|master's|ph\.?d|mba|degree|gpa|diploma|coursework|b\.tech|m\.tech|[bm]\.s\.|[bm]\.sc)/i;

  function compile(vocab, jobs, cal) {
    const rx = (p) => new RegExp(p, "i");
    const terms = vocab.terms.map((t) => ({ ...t, rx: rx(t.pattern) }));
    const byId = Object.fromEntries(terms.map((t) => [t.id, t]));
    const df = {};
    for (const j of jobs) for (const t of j.terms) df[t] = (df[t] || 0) + 1;
    const idf = {};
    for (const t of terms) idf[t.id] = Math.log((jobs.length + 1) / ((df[t.id] || 0) + 1)) + 1;
    return {
      terms, byId, idf, jobs, cal: cal || null,
      families: vocab.families.map((f) => ({ ...f, rx: new RegExp(f.pattern, "gi") })),
    };
  }

  function monthIndex(m) {
    if (!m) return 0;
    return MONTHS.split("|").indexOf(m.toLowerCase().slice(0, 3)) / 12;
  }

  // Years worked = union of date ranges like "Jan 2019 – Present", skipping degrees and capped at today.
  function experienceYears(text, now) { return experience(text, now).years; }

  // {years, known}: known = the résumé has job date ranges or says "N years of experience". Public or
  // anonymised résumés often have neither, and 0 then means "unknown", not "no experience".
  function experience(text, now) {
    const today = now || new Date();
    const nowYears = today.getFullYear() + today.getMonth() / 12;
    const spans = [];
    let prevEnd = 0;
    for (const m of text.matchAll(RANGE)) {
      // Look back over this line only, at most 100 chars and not past the previous range (PDF text
      // sometimes loses its line breaks).
      const from = Math.max(text.lastIndexOf("\n", m.index) + 1, m.index - 100, prevEnd);
      prevEnd = m.index + m[0].length;
      if (EDUCATION.test(text.slice(from, m.index))) continue;
      const start = +m[2] + monthIndex(m[1]);
      const end = Math.min(nowYears, m[3] ? nowYears : +m[5] + monthIndex(m[4]) + 1 / 12);
      if (end > start && end - start < 45) spans.push([start, end]);
    }
    spans.sort((a, b) => a[0] - b[0]);
    let total = 0, cur = null;
    for (const s of spans) {
      if (!cur || s[0] > cur[1]) { if (cur) total += cur[1] - cur[0]; cur = s.slice(); }
      else cur[1] = Math.max(cur[1], s[1]);
    }
    if (cur) total += cur[1] - cur[0];
    const stated = [...text.matchAll(/(\d{1,2})\+?\s*years?\s+(?:of\s+)?(?:\w+\s+){0,3}experience/gi)]
      .map((m) => +m[1]).filter((n) => n <= 40);
    // A stated "N years of experience" beats date ranges, which can include school.
    const best = stated.length ? Math.max(...stated) : total;
    return { years: Math.round(best * 2) / 2, known: stated.length > 0 || spans.length > 0 };
  }

  function analyzeResume(ctx, text, now) {
    const terms = new Set(ctx.terms.filter((t) => t.rx.test(text)).map((t) => t.id));
    // Family affinity: job-title mentions (strong) + skill terms in that family's categories.
    const raw = {};
    // The family named first in the headline counts most: "Regulatory Affairs Specialist" at the top
    // says more than a skills list.
    const headline = text.split("\n").map((l) => l.trim()).filter(Boolean).slice(0, 2).join(" ");
    let lead = null, leadAt = Infinity;
    for (const f of ctx.families) {
      const at = headline.search(new RegExp(f.pattern, "i"));
      if (at >= 0 && at < leadAt) { lead = f.id; leadAt = at; }
    }
    for (const f of ctx.families) {
      const titleHits = (text.match(f.rx) || []).length;
      const headlineBonus = f.id === lead ? 8 : 0;
      const cats = FAMILY_CATS[f.id];
      const skillHits = ctx.terms.filter((t) => terms.has(t.id) &&
        (cats.includes(t.cat) || (f.id === "DOC" && DOC_TERMS.includes(t.id)))).length;
      raw[f.id] = Math.min(titleHits, 6) * 2 + headlineBonus + skillHits;
    }
    const top = Math.max(1, ...Object.values(raw));
    const affinity = Object.fromEntries(Object.entries(raw).map(([k, v]) => [k, v / top]));
    const exp = experience(text, now);
    return { terms, affinity, years: exp.years, yearsKnown: exp.known, chars: text.length };
  }

  function expPoints(job, years) {
    const need = job.years != null ? job.years : LEVEL_YEARS[job.seniority] || 0;
    const gap = need - years;
    if (job.seniority === "director" && years < 8) return 0;
    if (gap <= 0) return job.seniority === "entry" && years - need > 3 ? 5 : 15;
    return gap <= 1 ? 11 : gap <= 2 ? 6 : 0;
  }

  function scoreJob(ctx, profile, job) {
    let have = 0, total = 0;
    const matched = [], missing = [];
    let jobInd = 0, sharedInd = 0;
    for (const id of job.terms) {
      const t = ctx.byId[id];
      if (!t) continue;
      if (t.cat === "industry") { jobInd++; if (profile.terms.has(id)) sharedInd++; continue; }
      total += ctx.idf[id];
      if (profile.terms.has(id)) { have += ctx.idf[id]; matched.push(id); } else missing.push(id);
    }
    const skills = have / (total + SMOOTH);
    const industry = jobInd ? (sharedInd ? 10 : 2) : 6;
    const family = (profile.affinity[job.family] || 0) * 20;
    const score = Math.round(skills * 55 + family + expPoints(job, profile.years) + industry);
    const byIdf = (a, b) => ctx.idf[b] - ctx.idf[a];
    return { score: Math.min(100, score), matched: matched.sort(byIdf), missing: missing.sort(byIdf) };
  }

  function scoreAll(ctx, profile) {
    const out = new Map();
    for (const j of ctx.jobs) out.set(j.id, scoreJob(ctx, profile, j));
    return out;
  }

  // Past H-1B filings or cap-exempt employer, and the posting doesn't rule visa holders out.
  function hasSponsorTrack(j) { return (j.h1b_ra_qa > 0 || j.cap_exempt) && !(j.restrict && j.restrict.length); }

  // Advice on widening the search. `jobs` = jobs passing the current filters, `all` = every job.
  function tips(ctx, profile, scores, jobs, all, filters, familyLabels) {
    const out = [];
    const strong = (j) => scores.get(j.id).score >= STRONG;
    const fits = (j) => scores.get(j.id).score >= WEAK;
    const shown = jobs.filter(fits);

    // 1. Skills that keep showing up in near-misses.
    const near = jobs.filter((j) => { const s = scores.get(j.id).score;
      return (profile.affinity[j.family] || 0) >= 0.5 && s >= WEAK - 10 && s < STRONG + 8; });
    const counts = {};
    for (const j of near) for (const id of scores.get(j.id).missing) if (!SKIP_TIP.has(id)) counts[id] = (counts[id] || 0) + 1;
    const gaps = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 6)
      .filter(([, n]) => n >= Math.max(2, near.length * 0.12));
    if (gaps.length) out.push({
      kind: "skills",
      title: "Add these if you have them",
      body: "Your near-matches keep asking for these, but your résumé doesn't name them. ATS filters match exact words.",
      chips: gaps.map(([id, n]) => ({ label: ctx.byId[id].label, n, of: near.length })),
    });

    // 2. Adjacent job families.
    for (const [fam, label] of Object.entries(familyLabels)) {
      if (filters.families.has(fam) && (profile.affinity[fam] || 0) >= 0.5) continue;
      const n = all.filter((j) => j.family === fam && fits(j)).length;
      if (n >= 3 && (profile.affinity[fam] || 0) < 0.5) out.push({
        kind: "family", title: `${n} ${label} roles fit you too`,
        body: "Search those titles as well.",
      });
    }

    // 3. Senior titles within reach.
    const senior = jobs.filter((j) => j.seniority === "senior" && j.years != null && j.years <= profile.years && fits(j));
    if (senior.length >= 2) out.push({
      kind: "level", title: `${senior.length} "Senior" roles need ${profile.years} years or less`,
      body: "Go by the years asked, not the title.",
    });

    // 4. Location / remote.
    if (filters.state) {
      const elsewhere = all.filter((j) => j.state !== filters.state && fits(j)).length;
      if (elsewhere) out.push({ kind: "location", title: `${elsewhere} more matches outside ${filters.state}`,
        body: "Try a nearby state or remote." });
    }
    if (filters.remote) {
      const onsite = all.filter((j) => !j.remote && fits(j)).length;
      if (onsite) out.push({ kind: "location", title: `${onsite} on-site matches hidden`,
        body: "Most RA/QA roles are on-site. Turn off remote-only to see them." });
    }

    // 5. Sponsorship.
    const sponsorStrong = shown.filter(hasSponsorTrack).length;
    if (!filters.h1b && sponsorStrong) out.push({ kind: "visa", title: `${sponsorStrong} matches have an H-1B record`,
      body: "Need sponsorship? Turn on the H-1B filter." });

    // 6. Industry carry-over.
    const ind = ctx.terms.filter((t) => t.cat === "industry");
    const yours = ind.filter((t) => profile.terms.has(t.id)).map((t) => t.id);
    if (yours.length) {
      for (const t of ind) {
        if (yours.includes(t.id) || t.id === "clinical") continue;
        const n = all.filter((j) => j.terms.includes(t.id) && !j.terms.some((x) => yours.includes(x)) && fits(j)).length;
        if (n >= 3) { out.push({ kind: "industry", title: `${n} matches in ${t.label.toLowerCase()}`,
          body: "QMS, CAPA and audit skills carry over." }); break; }
      }
    }
    return out;
  }

  // ---------- semantic layer ----------
  // Tested on 398 LLM-labeled jobs (tools/eval_rank.mjs): keyword alone Spearman 0.50; gte-small max chunk
  // similarity blended 50/50 with keyword 0.62 (bootstrap 95% CI of the gain +0.06..+0.20).
  const EMBED_MODEL = "Xenova/gte-base", CHUNK_WORDS = 180;
  // Weights on per-résumé z-scores, picked by 5-fold cross-validation over 40 labeled résumés
  // (tools/results/general_eval.md). NO_SEMANTIC is used until the model loads, or if it can't.
  const BLEND = { keyword: 0.3, lexical: 0.3, semantic: 0.4 };
  const BLEND_NO_SEMANTIC = { keyword: 0.3, lexical: 0.7 };

  // ---------- TF-IDF (same as scikit-learn: lowercase, \w\w+ tokens, English stop words dropped before
  // bigrams, sublinear tf, smooth idf, l2-normalised) ----------
  function lexTerms(text, stop) {
    const t = (String(text || "").toLowerCase().match(/[\p{L}\p{N}_]{2,}/gu) || []).filter((w) => !stop.has(w));
    const out = t.slice();
    for (let i = 0; i + 1 < t.length; i++) out.push(t[i] + " " + t[i + 1]);
    return out;
  }

  // docs: [{id, text}] -> {meta: {jobs, terms, idf, off}, post: Uint16Array (job index), wt: Uint16Array (×65535)}
  function buildLexical(docs, stopWords, minDf = 2) {
    const stop = new Set(stopWords), df = new Map();
    const counts = docs.map((d) => {
      const c = new Map();
      for (const t of lexTerms(d.text, stop)) c.set(t, (c.get(t) || 0) + 1);
      for (const t of c.keys()) df.set(t, (df.get(t) || 0) + 1);
      return c;
    });
    const terms = [...df.keys()].filter((t) => df.get(t) >= minDf).sort();
    const index = new Map(terms.map((t, i) => [t, i]));
    const idf = terms.map((t) => Math.log((1 + docs.length) / (1 + df.get(t))) + 1);
    const lists = terms.map(() => []);
    counts.forEach((c, j) => {
      const v = [];
      for (const [t, n] of c) if (index.has(t)) v.push([index.get(t), (1 + Math.log(n)) * idf[index.get(t)]]);
      const norm = Math.sqrt(v.reduce((a, [, w]) => a + w * w, 0)) || 1;
      for (const [i, w] of v) lists[i].push([j, w / norm]);
    });
    const off = [0];
    for (const l of lists) off.push(off[off.length - 1] + l.length);
    const post = new Uint16Array(off[off.length - 1]), wt = new Uint16Array(post.length);
    lists.forEach((l, i) => l.forEach(([j, w], k) => { post[off[i] + k] = j; wt[off[i] + k] = Math.round(w * 65535); }));
    return { meta: { jobs: docs.map((d) => d.id), terms, idf: idf.map((x) => +x.toFixed(5)), off, stop: stopWords }, post, wt };
  }

  // meta from lexical.json, buf = lexical.bin (post then wt) -> searchable index
  function loadLexical(meta, buf) {
    const n = meta.off[meta.off.length - 1];
    return { ...meta, index: new Map(meta.terms.map((t, i) => [t, i])), stopSet: new Set(meta.stop),
      post: new Uint16Array(buf, 0, n), wt: new Uint16Array(buf, n * 2, n) };
  }

  // Cosine between the résumé and every job -> Map(job_id -> similarity)
  function lexicalSims(lex, text) {
    const tf = new Map();
    for (const t of lexTerms(text, lex.stopSet)) {
      const i = lex.index.get(t);
      if (i !== undefined) tf.set(i, (tf.get(i) || 0) + 1);
    }
    const q = [...tf].map(([i, n]) => [i, (1 + Math.log(n)) * lex.idf[i]]);
    const norm = Math.sqrt(q.reduce((a, [, w]) => a + w * w, 0)) || 1;
    const sims = new Float64Array(lex.jobs.length);
    for (const [i, w] of q)
      for (let p = lex.off[i]; p < lex.off[i + 1]; p++) sims[lex.post[p]] += (w / norm) * (lex.wt[p] / 65535);
    return new Map(lex.jobs.map((id, k) => [id, sims[k]]));
  }

  function chunks(text) {
    const w = String(text || "").split(/\s+/).filter(Boolean), out = [];
    for (let i = 0; i < w.length; i += CHUNK_WORDS) out.push(w.slice(i, i + CHUNK_WORDS).join(" "));
    return out.length ? out : [""];
  }

  // Best cosine between any résumé chunk and any of the job's chunks (rows of Int16 unit vectors × 32767).
  function maxSim(resumeVecs, rows, first, count, dim) {
    let best = -1;
    for (let r = first; r < first + count; r++) {
      const off = r * dim;
      for (const v of resumeVecs) {
        let s = 0;
        for (let i = 0; i < dim; i++) s += v[i] * rows[off + i];
        if (s > best) best = s;
      }
    }
    return best / 32767;
  }

  // Blend similarity signals into keyword scores in place: weighted sum of z-scores (BLEND), then map the
  // result back onto the keyword scores' own mean and spread, so the Strong / Worth-a-look cut-offs keep
  // their meaning (a plain average would squeeze scores toward the middle).
  // signals: {lexical?: Map, semantic?: Map}
  function blendSignals(scores, signals) {
    const names = Object.keys(signals).filter((k) => signals[k]);
    if (!names.length) return;
    const weights = names.includes("semantic") ? BLEND : BLEND_NO_SEMANTIC;
    const ids = [...scores.keys()].filter((id) => names.every((k) => signals[k].has(id)));
    if (ids.length < 2) return;
    const stats = (xs) => { const m = xs.reduce((a, b) => a + b, 0) / xs.length;
      return [m, Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length) || 1]; };
    const kw = ids.map((id) => scores.get(id).keyword ?? scores.get(id).score);
    const [mk, sk] = stats(kw);
    const mix = kw.map((x) => weights.keyword * (x - mk) / sk);
    for (const k of names) {
      const v = ids.map((id) => signals[k].get(id)), [m, s] = stats(v);
      v.forEach((x, i) => { mix[i] += (weights[k] || 0) * (x - m) / s; });
    }
    const [mm, sm] = stats(mix);
    ids.forEach((id, i) => {
      const s = scores.get(id);
      s.keyword = kw[i];
      s.raw = mk + sk * (mix[i] - mm) / sm;           // unrounded, unclamped: used for ranking and calibration
      s.score = Math.max(0, Math.min(100, Math.round(s.raw)));
    });
  }

  // After the blend, take points off when the posting asks for more years than the résumé shows. The text
  // signals ignore seniority, so a strong text match a year or two short could still score 86. Size picked by
  // 5-fold CV on the eval set (tools/results/exp_penalty.md); larger penalties ranked worse against the labels.
  // Skipped when the résumé's years are unknown (no dates, no "N years"), unless the visitor typed them in.
  const EXP_PENALTY = { perYear: 15, cap: 10, fromTitle: 0.5, director: 15 };
  function experienceAdjust(ctx, profile, scores) {
    if (!profile.yearsKnown) return;
    for (const j of ctx.jobs) {
      const s = scores.get(j.id);
      if (!s || s.expPenalty != null) continue;
      const need = j.years != null ? j.years : LEVEL_YEARS[j.seniority] || 0;
      const gap = need - profile.years;
      let pen = Math.min(EXP_PENALTY.cap, EXP_PENALTY.perYear * Math.max(0, gap)) * (j.years != null ? 1 : EXP_PENALTY.fromTitle);
      if (j.seniority === "director" && gap > 2) pen = Math.max(pen, EXP_PENALTY.director);
      s.expPenalty = pen;
      s.raw = (s.raw ?? s.score) - pen;
      s.score = Math.max(0, Math.min(100, Math.round(s.raw)));
    }
  }

  // Map the blended score onto the rubric's scale (85+ strong, 70–84 good, 50–69 partial, under 50 poor) with
  // the table in site/calibration.json, fit on the 1,860 labeled pairs (tools/eval/calibrate.py). Monotone, so
  // the ranking doesn't change; it only makes the number mean the same thing as a hiring manager's rating.
  // cal = {x: [raw scores, ascending], y: [rubric scores]}; linear between points, flat beyond the ends.
  function calibrate(cal, raw) {
    if (!cal) return raw;
    const { x, y } = cal;
    if (raw <= x[0]) return y[0];
    if (raw >= x[x.length - 1]) return y[y.length - 1];
    let i = 1;
    while (x[i] < raw) i++;
    return y[i - 1] + (y[i] - y[i - 1]) * (raw - x[i - 1]) / (x[i] - x[i - 1]);
  }

  // Everything after scoreAll: blend in the similarity signals, adjust for experience, calibrate.
  // s.rank (continuous) orders jobs; s.score (0–100 integer, rubric scale) is what visitors see.
  function finalize(ctx, profile, scores, signals) {
    signals = signals || {};
    blendSignals(scores, signals);
    experienceAdjust(ctx, profile, scores);
    // calibration.json has one table per blend: "full" once the AI model has loaded, "lexical" before that.
    const cal = ctx.cal && (signals.semantic ? ctx.cal.full : ctx.cal.lexical);
    for (const s of scores.values()) {
      s.rank = s.raw ?? s.score;
      s.score = Math.max(0, Math.min(100, Math.round(calibrate(cal, s.rank))));
    }
  }

  const M = { STRONG, WEAK, compile, analyzeResume, experienceYears, scoreJob, scoreAll, tips, hasSponsorTrack,
    LEVEL_YEARS, EMBED_MODEL, chunks, maxSim, blendSignals, experienceAdjust, calibrate, finalize,
    buildLexical, loadLexical, lexicalSims };
  if (typeof module !== "undefined" && module.exports) module.exports = M; else root.Match = M;
})(typeof self !== "undefined" ? self : this);   // window, a worker, or Node
