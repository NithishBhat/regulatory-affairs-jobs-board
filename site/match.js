// Résumé ↔ job matching. Pure functions (no DOM) so they run in the browser and in Node tests.
(function (root) {
  "use strict";

  const STRONG = 68, WEAK = 50;
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

  function compile(vocab, jobs) {
    const rx = (p) => new RegExp(p, "i");
    const terms = vocab.terms.map((t) => ({ ...t, rx: rx(t.pattern) }));
    const byId = Object.fromEntries(terms.map((t) => [t.id, t]));
    const df = {};
    for (const j of jobs) for (const t of j.terms) df[t] = (df[t] || 0) + 1;
    const idf = {};
    for (const t of terms) idf[t.id] = Math.log((jobs.length + 1) / ((df[t.id] || 0) + 1)) + 1;
    return {
      terms, byId, idf, jobs,
      families: vocab.families.map((f) => ({ ...f, rx: new RegExp(f.pattern, "gi") })),
    };
  }

  function monthIndex(m) {
    if (!m) return 0;
    return MONTHS.split("|").indexOf(m.toLowerCase().slice(0, 3)) / 12;
  }

  // Years worked = union of date ranges like "Jan 2019 – Present".
  function experienceYears(text, now) {
    const today = now || new Date();
    const spans = [];
    for (const m of text.matchAll(RANGE)) {
      const start = +m[2] + monthIndex(m[1]);
      const end = m[3] ? today.getFullYear() + today.getMonth() / 12 : +m[5] + monthIndex(m[4]) + 1 / 12;
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
    return Math.round(best * 2) / 2;
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
    return { terms, affinity, years: experienceYears(text, now), chars: text.length };
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

  function hasSponsorTrack(j) { return j.h1b_ra_qa > 0 || j.cap_exempt; }

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
      title: "Terms your closest matches ask for",
      body: "These come up often in jobs just below or near your strong matches, and your résumé doesn't mention them. " +
        "If you have this experience, name it in plain words. Applicant tracking systems match on the exact term.",
      chips: gaps.map(([id, n]) => ({ label: ctx.byId[id].label, n, of: near.length })),
    });

    // 2. Adjacent job families.
    for (const [fam, label] of Object.entries(familyLabels)) {
      if (filters.families.has(fam) && (profile.affinity[fam] || 0) >= 0.5) continue;
      const n = all.filter((j) => j.family === fam && fits(j)).length;
      if (n >= 3 && (profile.affinity[fam] || 0) < 0.5) out.push({
        kind: "family", title: `${n} ${label} roles also fit`,
        body: `Your skills overlap with ${label} postings. Search for those titles too, and use the same keywords in your headline.`,
      });
    }

    // 3. Senior titles within reach.
    const senior = jobs.filter((j) => j.seniority === "senior" && j.years != null && j.years <= profile.years && fits(j));
    if (senior.length >= 2) out.push({
      kind: "level", title: `${senior.length} "Senior" roles ask for no more than your ${profile.years} years`,
      body: "Don't rule out a posting because of its title. Go by the years and skills it actually asks for.",
    });

    // 4. Location / remote.
    if (filters.state) {
      const elsewhere = all.filter((j) => j.state !== filters.state && fits(j)).length;
      if (elsewhere) out.push({ kind: "location", title: `${elsewhere} more matches outside ${filters.state}`,
        body: "Relocation-friendly employers, especially large manufacturers, often post in several states. Try a nearby state or turn on remote." });
    }
    if (filters.remote) {
      const onsite = all.filter((j) => !j.remote && fits(j)).length;
      if (onsite) out.push({ kind: "location", title: `${onsite} on-site or hybrid matches hidden`,
        body: "Most RA/QA work at manufacturers is on-site. Turning off remote-only greatly widens the pool." });
    }

    // 5. Sponsorship.
    const sponsorStrong = shown.filter(hasSponsorTrack).length;
    if (!filters.h1b && sponsorStrong) out.push({ kind: "visa", title: `${sponsorStrong} of your matches have an H-1B track record`,
      body: "If you need sponsorship, turn on \"H-1B track record only\". Universities and hospitals can file cap-exempt H-1Bs year-round." });

    // 6. Industry carry-over.
    const ind = ctx.terms.filter((t) => t.cat === "industry");
    const yours = ind.filter((t) => profile.terms.has(t.id)).map((t) => t.id);
    if (yours.length) {
      for (const t of ind) {
        if (yours.includes(t.id) || t.id === "clinical") continue;
        const n = all.filter((j) => j.terms.includes(t.id) && !j.terms.some((x) => yours.includes(x)) && fits(j)).length;
        if (n >= 3) { out.push({ kind: "industry", title: `Your skills carry over to ${t.label.toLowerCase()} (${n} matches)`,
          body: "QMS, CAPA and audit experience transfer across regulated industries. In your summary, state the frameworks you know (e.g. GMP, ISO)." }); break; }
      }
    }
    return out;
  }

  const M = { STRONG, WEAK, compile, analyzeResume, experienceYears, scoreJob, scoreAll, tips, hasSponsorTrack };
  if (typeof module !== "undefined" && module.exports) module.exports = M; else root.Match = M;
})(this);
