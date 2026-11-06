// UI: load jobs, read the résumé locally, filter, render.
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const PAGE = 40, TIPS_SHOWN = 3;
  const LEVELS = { entry: "Entry", mid: "Mid", senior: "Senior", manager: "Manager / Lead", director: "Director+" };
  const US = /^(?:us|usa|united states(?: of america)?)$/i;
  const GENERIC = new Set(["fda", "audit", "sop"]); // nearly every posting has these; not worth a chip
  // Work-authorization limits stated in the posting (export.py), most restrictive first.
  const RESTRICT = {
    citizen: ["US citizens / green card only", "The posting says only US citizens (or green card holders) can apply"],
    clearance: ["Security clearance", "Needs a US security clearance, which requires US citizenship"],
    export: ["US persons only", "Export-controlled work (ITAR/EAR): US citizens, green card holders, asylees or refugees"],
    no_sponsor: ["No visa sponsorship", "The posting says it won't sponsor a work visa (H-1B etc.)"],
  };
  const restricted = (j) => !!(j.restrict && j.restrict.length);
  function limitBadge(j) {
    const first = Object.keys(RESTRICT).find((k) => j.restrict.includes(k));
    const why = j.restrict.map((k) => RESTRICT[k][1]).join(". ");
    return `<span class="badge limit" title="${esc(why)}">${RESTRICT[first][0]}</span>`;
  }
  const TRANSFORMERS = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
  const SEMANTIC_ON = '<span class="sem on" title="Keyword and word-overlap match blended with meaning-based similarity, computed in your browser">· AI matching on</span>';
  const S = { ctx: null, jobs: [], labels: {}, profile: null, scores: null, tab: "all", limit: PAGE,
    text: "", years: null, showSkills: false, showAllTips: false, sims: null, simsText: "", semantic: null, semRun: 0,
    lex: null, lexSims: null, lexText: "", newIds: new Set(), hasPrevVisit: false };

  // ---------- this browser's own lists (localStorage; nothing leaves the browser) ----------
  const store = {
    get(k, d) { try { const v = localStorage.getItem("rqa." + k); return v ? JSON.parse(v) : d; } catch { return d; } },
    set(k, v) { try { localStorage.setItem("rqa." + k, JSON.stringify(v)); } catch { /* private mode: lists last this visit */ } },
  };
  // mine: {job_id: {status: "saved" | "applied", at: ISO date, job: {title, company, url, location}}}
  const MY = { mine: store.get("mine", {}), hidden: new Set(store.get("hiddenJobs", [])),
    hiddenCo: new Set(store.get("hiddenCompanies", [])) };
  const coKey = (c) => String(c || "").trim().toLowerCase();
  const isHidden = (j) => MY.hidden.has(j.id) || MY.hiddenCo.has(coKey(j.company));
  function saveMine() {
    store.set("mine", MY.mine);
    store.set("hiddenJobs", [...MY.hidden]);
    store.set("hiddenCompanies", [...MY.hiddenCo]);
  }

  // "New since your last visit": job ids seen on the previous visit (a visit = 6+ hours after the last one).
  const VISIT_GAP = 6 * 3600e3;
  function trackVisit(ids) {
    const v = store.get("visits", null);
    let prev = null;
    if (!v) store.set("visits", { prev: null, cur: ids, at: Date.now() });
    else if (Date.now() - v.at > VISIT_GAP) { prev = v.cur; store.set("visits", { prev: v.cur, cur: ids, at: Date.now() }); }
    else prev = v.prev;
    if (!prev) return;
    const seen = new Set(prev);
    S.hasPrevVisit = true;
    S.newIds = new Set(ids.filter((id) => !seen.has(id)));
  }

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const daysAgo = (d) => Math.max(0, Math.round((Date.now() - new Date(d + "T12:00:00")) / 864e5));

  async function init() {
    const [vocab, data] = await Promise.all([
      fetch("vocab.json").then((r) => r.json()),
      fetch("jobs.json", { cache: "no-cache" }).then((r) => r.json()),
    ]);
    S.jobs = data.jobs;
    trackVisit(S.jobs.map((j) => j.id));
    S.ctx = Match.compile(vocab, S.jobs);
    S.labels = Object.fromEntries(vocab.families.map((f) => [f.id, f.label]));
    $("trustCount").textContent = data.count.toLocaleString();
    $("nudgeBtn").addEventListener("click", () => $("file").click());
    $("meta").textContent = `${data.count.toLocaleString()} open roles · updated ${new Date(data.generated).toLocaleDateString(undefined, { month: "short", day: "numeric" })}`;

    $("families").innerHTML = vocab.families.map((f) =>
      `<label class="check"><input type="checkbox" value="${f.id}" checked> ${esc(f.label)} <span class="count">${S.jobs.filter((j) => j.family === f.id).length}</span></label>`).join("");
    $("level").insertAdjacentHTML("beforeend", Object.entries(LEVELS).map(([k, v]) => `<option value="${k}">${v}</option>`).join(""));
    const states = [...new Set(S.jobs.map((j) => j.state).filter(Boolean))].sort();
    $("state").insertAdjacentHTML("beforeend", states.map((s) => `<option>${s}</option>`).join(""));

    $("newCount").textContent = S.newIds.size;
    $("newOnlyBox").hidden = !S.hasPrevVisit;
    $("entryCount").textContent = S.jobs.filter(isEntry).length;
    $("visaOkCount").textContent = S.jobs.filter((j) => !restricted(j)).length;

    document.querySelectorAll(".filters input, .filters select").forEach((el) =>
      el.addEventListener(el.type === "search" ? "input" : "change", () => { S.limit = PAGE; render(); }));
    wireUpload();
    render();
    loadLexical();
  }

  // ---------- résumé input ----------
  function wireUpload() {
    const drop = $("drop");
    $("file").addEventListener("change", (e) => e.target.files[0] && readFile(e.target.files[0]));
    ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove("over")));
    drop.addEventListener("drop", (e) => { e.preventDefault(); e.dataTransfer.files[0] && readFile(e.dataTransfer.files[0]); });
    $("pasteBtn").addEventListener("click", () => { $("pasteBox").hidden = false; $("pasteBtn").hidden = true; $("paste").focus(); });
    $("paste").addEventListener("input", () => { S.text = $("paste").value; $("matchBtn").disabled = S.text.trim().length < 200; });
    $("matchBtn").addEventListener("click", () => { S.years = null; runMatch(); });

    // Profile summary controls (rendered later), delegated.
    $("profile").addEventListener("click", (e) => {
      const act = e.target.dataset.act;
      if (act === "years") { e.target.outerHTML = `<input type="number" id="yearsInput" class="inline" min="0" max="40" step="0.5" value="${S.profile.years}"> years`; $("yearsInput").focus(); }
      if (act === "skills") { S.showSkills = !S.showSkills; renderProfile(); }
      if (act === "clear") clearResume();
    });
    $("profile").addEventListener("change", (e) => {
      if (e.target.id === "yearsInput") { S.years = Math.max(0, +e.target.value || 0); runMatch(); }
    });
    $("tips").addEventListener("click", (e) => {
      if (e.target.dataset.act === "tips") { S.showAllTips = !S.showAllTips; render(); }
    });
  }

  async function readFile(f) {
    status(`Reading ${f.name}…`);
    try {
      const name = f.name.toLowerCase();
      if (name.endsWith(".pdf")) S.text = await pdfText(await f.arrayBuffer());
      else if (name.endsWith(".docx")) S.text = (await mammoth.extractRawText({ arrayBuffer: await f.arrayBuffer() })).value;
      else S.text = await f.text();
      if (S.text.trim().length < 200) return status("Couldn't find much text in that file. If it's a scanned PDF, paste the text instead.", true);
      S.years = null;
      runMatch();
    } catch (err) {
      status("Couldn't read that file. Try another format or paste the text.", true);
      console.error(err);
    }
  }

  async function pdfText(buf) {
    pdfjsLib.GlobalWorkerOptions.workerSrc = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
    const doc = await pdfjsLib.getDocument({ data: buf }).promise;
    const pages = [];
    for (let i = 1; i <= doc.numPages; i++) {
      const c = await (await doc.getPage(i)).getTextContent();
      pages.push(c.items.map((it) => it.str + (it.hasEOL ? "\n" : " ")).join(""));
    }
    return pages.join("\n");
  }

  // S.years = null means "use what the résumé says"; a number is the visitor's correction.
  function runMatch() {
    const first = !S.profile;
    S.profile = Match.analyzeResume(S.ctx, S.text);
    if (S.years != null) S.profile.years = S.years;
    applySignals();
    if (!(S.sims && S.simsText === S.text)) refineSemantic();   // a years edit reuses the similarities
    S.tab = "strong"; S.limit = PAGE;
    document.body.classList.add("matched");
    status("");
    renderProfile();
    render();
    if (first) $("profile").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // Keyword scores blended with TF-IDF and (once the model has loaded) embedding similarity.
  function applySignals() {
    S.scores = Match.scoreAll(S.ctx, S.profile);
    if (S.lex && S.lexText !== S.text) Object.assign(S, { lexSims: Match.lexicalSims(S.lex, S.text), lexText: S.text });
    Match.blendSignals(S.scores, { lexical: S.lexText === S.text ? S.lexSims : null,
      semantic: S.simsText === S.text ? S.sims : null });
  }

  // TF-IDF index (~5 MB, no model): loads with the page, so it is normally ready before a résumé is.
  function loadLexical() {
    Promise.all([fetch("lexical.json").then((r) => r.json()), fetch("lexical.bin").then((r) => r.arrayBuffer())])
      .then(([meta, buf]) => {
        S.lex = Match.loadLexical(meta, buf);
        if (S.profile) { applySignals(); render(); }
      })
      .catch((err) => console.error(err));             // keyword (+ semantic) matching still works
  }

  // ---------- semantic layer: keyword results show at once, then the model refines them ----------
  // Download progress, summed over the model's files; only the note's text is updated, not the page.
  const dl = new Map();
  let dlShown = 0;
  function onModelProgress(e) {
    if (e.status === "progress" && e.total) dl.set(e.file, [e.loaded, e.total]);
    if (Date.now() - dlShown < 250) return;
    dlShown = Date.now();
    const el = document.querySelector("#profile .sem");
    if (el && S.semantic === "loading") el.textContent = loadingText();
  }
  function loadingText() {
    let loaded = 0, total = 0;
    for (const [l, t] of dl.values()) { loaded += l; total += t; }
    const mb = (x) => Math.round(x / 1e6);
    return total > 5e6 && loaded < total
      ? `· improving scores with AI · ${mb(loaded)} of ${mb(total)} MB (one-time download)`
      : "· improving scores with AI…";
  }

  let semantic = null;
  function loadSemantic() {
    semantic = semantic || Promise.all([
      import(TRANSFORMERS),
      fetch("vectors.json").then((r) => r.json()),
      fetch("vectors.bin").then((r) => r.arrayBuffer()),
    ]).then(async ([tf, meta, buf]) => ({
      meta, rows: new Int16Array(buf),
      embed: await tf.pipeline("feature-extraction", meta.model, { dtype: "q8", progress_callback: onModelProgress }),
    }));
    semantic.catch(() => { semantic = null; });       // allow a retry on the next résumé
    return semantic;
  }

  async function refineSemantic() {
    const run = ++S.semRun, text = S.text;
    S.sims = null; S.semantic = "loading";
    try {
      const sem = await loadSemantic();
      const vecs = (await sem.embed(Match.chunks(text), { pooling: "mean", normalize: true })).tolist();
      const sims = new Map();
      for (const [id, [first, count]] of Object.entries(sem.meta.index))
        sims.set(id, Match.maxSim(vecs, sem.rows, first, count, sem.meta.dim));
      if (run !== S.semRun || !S.scores) return;                // résumé changed or cleared meanwhile
      Object.assign(S, { sims, simsText: text, semantic: "on" });
      applySignals();
    } catch (err) {
      console.error(err);
      if (run !== S.semRun) return;
      S.semantic = "off";
    }
    if (S.profile) { renderProfile(); render(); }
  }

  function clearResume() {
    Object.assign(S, { profile: null, scores: null, text: "", years: null, tab: "all", limit: PAGE, showSkills: false,
      showAllTips: false, sims: null, simsText: "", lexSims: null, lexText: "", semantic: null, semRun: S.semRun + 1 });
    $("paste").value = ""; $("pasteBox").hidden = true; $("pasteBtn").hidden = false; $("file").value = "";
    $("matchBtn").disabled = true;
    $("profile").hidden = true; $("tips").hidden = true;
    document.body.classList.remove("matched");
    render();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function status(msg, bad) { $("status").textContent = msg; $("status").classList.toggle("bad", !!bad); }

  // ---------- filtering ----------
  function filters() {
    return {
      q: $("q").value.trim().toLowerCase(),
      families: new Set([...document.querySelectorAll("#families input:checked")].map((i) => i.value)),
      level: $("level").value, state: $("state").value, age: +$("age").value,
      remote: $("remote").checked, h1b: $("h1b").checked, entry: $("entry").checked, newOnly: $("newOnly").checked,
      visaOk: $("visaOk").checked,
    };
  }

  // Entry level / new grad: an entry title, or a non-senior title asking for at most 2 years.
  const isEntry = (j) => j.seniority === "entry" ||
    (j.years != null && j.years <= 2 && !["senior", "manager", "director"].includes(j.seniority));

  function applyFilters(f) {
    return S.jobs.filter((j) => !isHidden(j) &&
      (!f.entry || isEntry(j)) && (!f.newOnly || S.newIds.has(j.id)) &&
      f.families.has(j.family) && (!f.level || j.seniority === f.level) &&
      (!f.state || j.state === f.state || (j.remote && !j.state)) && (!f.remote || j.remote) &&
      (!f.h1b || Match.hasSponsorTrack(j)) && (!f.visaOk || !restricted(j)) && daysAgo(j.posted) <= f.age &&
      (!f.q || (j.title + " " + j.company).toLowerCase().includes(f.q)));
  }

  // "Piscataway, NJ, United States, US" -> "Piscataway, NJ"; messy ATS strings -> state only.
  function shortLoc(j) {
    const parts = (j.location || "").split(/[,>:|]/).map((p) => p.trim()).filter((p) => p && !US.test(p) && !/^\d/.test(p));
    const city = parts[0] || "";
    if (!j.state) return city.length && city.length < 30 ? city : "United States";
    return city && city !== j.state && city.length < 25 && !/[~\d]/.test(city) ? `${city}, ${j.state}` : j.state;
  }

  // ---------- rendering ----------
  function renderProfile() {
    const p = S.profile, labels = S.labels;
    const fam = Object.entries(p.affinity).sort((a, b) => b[1] - a[1]).filter(([, v]) => v >= 0.35).map(([k]) => labels[k]);
    const terms = [...p.terms].map((id) => S.ctx.byId[id].label);
    $("profile").hidden = false;
    $("profile").innerHTML = `
      <p>Matching you as <strong>${esc(fam.join(" + ") || "Quality")}</strong> with
        <button type="button" class="link" data-act="years" title="Not right? Click to change">${p.years} years</button> experience ·
        <button type="button" class="link" data-act="skills">${terms.length} skills found</button>
        ${S.semantic === "loading" ? `<span class="sem" title="Runs in your browser; your résumé is not uploaded">${loadingText()}</span>`
          : S.semantic === "on" ? SEMANTIC_ON : ""}
        <button type="button" class="link muted-link" data-act="clear">Use another résumé</button></p>
      ${S.showSkills ? `<div class="chips">${terms.map((t) => `<span class="chip ok">${esc(t)}</span>`).join("")}</div>` : ""}`;
  }

  function renderTips(list, f) {
    const all = Match.tips(S.ctx, S.profile, S.scores, list, S.jobs, f, S.labels);
    const t = S.showAllTips ? all : all.slice(0, TIPS_SHOWN);
    $("tips").hidden = !all.length;
    $("tips").innerHTML = `<ul class="tip-list">${t.map((x) => `
      <li class="tip"><strong>${esc(x.title)}</strong> <span class="muted">${esc(x.body)}</span>
      ${x.chips ? `<div class="chips">${x.chips.map((c) => `<span class="chip gap" title="asked for in ${c.n} of ${c.of} close matches">${esc(c.label)}</span>`).join("")}</div>` : ""}
      </li>`).join("")}</ul>
      ${all.length > TIPS_SHOWN ? `<button type="button" class="link small" data-act="tips">${S.showAllTips ? "Fewer tips" : `${all.length - TIPS_SHOWN} more tips`}</button>` : ""}`;
  }

  function render() {
    const f = filters();
    let list = applyFilters(f);
    const sc = (j) => S.scores.get(j.id).score;
    let tabs;
    if (S.scores) {
      list.sort((a, b) => sc(b) - sc(a));
      const strong = list.filter((j) => sc(j) >= Match.STRONG);
      const weak = list.filter((j) => sc(j) >= Match.WEAK && sc(j) < Match.STRONG);
      tabs = [["strong", "Strong matches", strong], ["weak", "Worth a look", weak], ["all", "All jobs", list]];
      renderTips(list, f);
    } else {
      tabs = [["all", "All jobs", list]];
    }
    const mine = myList();
    if (mine.length || S.tab === "mine") tabs.push(["mine", "My list", mine]);
    const current = tabs.find((t) => t[0] === S.tab) || tabs[0];
    $("tabs").innerHTML = tabs.map(([id, label, l]) =>
      `<button role="tab" aria-selected="${id === current[0]}" data-tab="${id}">${label} <span class="count">${l.length}</span></button>`).join("");
    $("tabs").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => { S.tab = b.dataset.tab; S.limit = PAGE; render(); }));

    $("nudge").hidden = !!S.scores;
    const rows = current[2];
    const active = current[0] === "mine" ? hideActive() : renderActive(f, list.length);
    $("jobs").innerHTML = rows.length ? rows.slice(0, S.limit).map(card).join("")
      : `<li class="empty">${current[0] === "mine" ? "Nothing saved yet. Click the ☆ on a job to keep it here."
        : active ? 'No jobs here with these filters. <button type="button" class="link" data-act="clear-filters">Clear all filters</button>'
        : "No jobs in this tab. Try the other tabs."}</li>`;
    $("more").hidden = rows.length <= S.limit;
    $("more").onclick = () => { S.limit += PAGE; render(); };
  }

  function card(j) {
    if (j.gone) {
      const m = MY.mine[j.id];
      return `<li class="job card is-mine gone" data-id="${esc(j.id)}">${avatar(j.company)}<div class="job-main">
        <div class="title-row"><a class="job-title" href="${esc(j.url)}" target="_blank" rel="noopener nofollow">${esc(j.title)}</a>
          <span class="badge mine ${m.status}">${m.status === "applied" ? "Applied" : "Saved"} ${fmtDate(m.at)}</span></div>
        <div class="job-sub"><strong>${esc(j.company)}</strong> · <span class="muted">No longer listed here</span></div></div>
        <div class="side">${actions(j, m)}</div></li>`;
    }
    const s = S.scores && S.scores.get(j.id);
    const age = daysAgo(j.posted);
    const role = j.family === "RA" ? "regulatory affairs" : j.family === "QC" ? "quality control" : "quality";
    const li = "https://www.linkedin.com/search/results/people/?keywords=" + encodeURIComponent(`${j.company} ${role} manager`);
    // A stated limit (citizens only, clearance, no sponsorship…) replaces the H-1B badge: the past filings
    // don't help with this role.
    const visa = restricted(j) ? limitBadge(j)
      : j.h1b_ra_qa > 0 ? `<span class="badge visa" title="${j.h1b_ra_qa} H-1B applications for RA/QA roles, DOL FY2024–26">H-1B sponsor</span>`
      : j.cap_exempt ? `<span class="badge visa soft" title="Universities and hospitals can usually sponsor H-1B cap-exempt">May be cap-exempt</span>` : "";
    const posted = age === 0 ? "today" : `${age}d ago`;
    const hire = `<a href="${li}" target="_blank" rel="noopener nofollow" title="LinkedIn people search">Find hiring team</a>`;
    let checks, missing = "";
    if (s) {
      // "How you match": skills, experience vs. what the posting asks, level. Symbol + colour, never colour alone.
      const total = s.matched.length + s.missing.length;
      const items = [];
      if (total) items.push(check(s.matched.length / total >= 0.5 ? "ok" : "warn", `${s.matched.length} of ${total} skills`,
        "Skills this posting asks for that your résumé names"));
      if (j.years != null) {
        const gap = j.years - S.profile.years;
        items.push(gap <= 0 ? check("ok", `Asks ${j.years}+ yrs`, `You have ${S.profile.years}`)
          : check(gap <= 1 ? "warn" : "no", `Asks ${j.years}+ yrs`, `You have ${S.profile.years}`));
      }
      items.push(`<span class="ck plain">${LEVELS[j.seniority]}</span>`);
      checks = items.join("");
      const gaps = s.missing.filter((id) => !GENERIC.has(id)).slice(0, 3);
      if (gaps.length) missing = `<div class="missing"><span class="muted">Missing:</span>
        ${gaps.map((id) => `<span class="chip gap" title="Asked for, not on your résumé">${esc(S.ctx.byId[id].label)}</span>`).join("")}</div>`;
    } else {
      checks = `<span class="ck plain">${LEVELS[j.seniority]}${j.years != null ? ` · ${j.years}+ yrs` : ""}</span>`;
    }
    const m = MY.mine[j.id];
    const status = m ? `<span class="badge mine ${m.status}">${m.status === "applied" ? "Applied" : "Saved"} ${fmtDate(m.at)}</span>` : "";
    const fresh = S.newIds.has(j.id) ? '<span class="badge new" title="Not listed on your last visit">New</span>' : "";
    const score = s ? `<div class="score ${s.score >= Match.STRONG ? "hi" : s.score >= Match.WEAK ? "mid" : "lo"}" title="Match score (0–100)">${s.score}</div>` : "";
    return `<li class="job card${m ? " is-mine" : ""}" data-id="${esc(j.id)}">
      ${avatar(j.company)}
      <div class="job-main">
        <div class="title-row"><a class="job-title" href="${esc(j.url)}" target="_blank" rel="noopener nofollow">${esc(j.title)}</a>${fresh}${status}</div>
        <div class="job-sub"><strong>${esc(j.company)}</strong> · ${esc(shortLoc(j))}${j.remote && !/remote/i.test(shortLoc(j)) ? ' · <span class="remote">Remote</span>' : ""} ${visa}</div>
        <div class="checks">${checks}<span class="ck plain">${posted}</span><span class="ck plain">${hire}</span></div>
        ${missing}
      </div>
      <div class="side">${score}${actions(j, m)}</div></li>`;
  }

  // Small icon buttons: save, mark applied, more (hide). Labels for screen readers and on hover.
  const ICON = {
    star: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3.5l2.6 5.3 5.9.9-4.3 4.1 1 5.8-5.2-2.7-5.2 2.7 1-5.8-4.3-4.1 5.9-.9z"/></svg>',
    check: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4.5 12.5l5 5 10-11"/></svg>',
    more: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="5" cy="12" r="1.6"/><circle cx="12" cy="12" r="1.6"/><circle cx="19" cy="12" r="1.6"/></svg>',
  };
  function actions(j, m) {
    const on = (st) => !!(m && m.status === st);
    const btn = (st, icon, label, onLabel) => `<button type="button" class="ico ${st}${on(st) ? " on" : ""}" data-do="${st}"
      aria-pressed="${on(st)}" title="${on(st) ? onLabel : label}" aria-label="${on(st) ? onLabel : label}">${ICON[icon]}</button>`;
    return `<div class="actions">
      ${btn("saved", "star", "Save", "Saved — click to remove")}
      ${btn("applied", "check", "Mark as applied", "Applied — click to undo")}
      <details class="hide-menu"><summary class="ico" title="More" aria-label="More options">${ICON.more}</summary>
        <div class="menu"><button type="button" data-do="hide-job">Hide this job</button>
          <button type="button" data-do="hide-co">Hide all from ${esc(j.company)}</button></div></details>
    </div>`;
  }

  // Company initials in a colour picked from the name, so the same company always looks the same.
  const LEGAL = /^(?:inc|llc|ltd|corp|corporation|co|company|plc|lp|the|group|holdings|usa|us|america|americas|na)$/i;
  function avatar(company) {
    const words = String(company || "?").replace(/[^\p{L}\p{N}\s]/gu, " ").split(/\s+/).filter((w) => w && !LEGAL.test(w));
    const initials = ((words[0] || "?")[0] + (words.length > 1 ? words[1][0] : (words[0] || "")[1] || "")).toUpperCase();
    let h = 0;
    for (const c of String(company)) h = (h * 31 + c.charCodeAt(0)) % 360;
    return `<div class="avatar" style="--h:${h}" aria-hidden="true">${esc(initials)}</div>`;
  }

  const fmtDate = (iso) => new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric" });

  // Saved / applied jobs: saved first, newest first; jobs no longer listed come from the saved snapshot.
  function myList() {
    const byId = new Map(S.jobs.map((j) => [j.id, j]));
    return Object.entries(MY.mine)
      .sort((a, b) => (a[1].status === b[1].status ? 0 : a[1].status === "saved" ? -1 : 1) || b[1].at.localeCompare(a[1].at))
      .map(([id, m]) => byId.get(id) || { ...m.job, id, gone: true });
  }

  function hideActive() { $("active").hidden = true; return 0; }

  // Close an open Hide menu when clicking anywhere else.
  document.addEventListener("click", (e) => {
    document.querySelectorAll(".hide-menu[open]").forEach((d) => { if (!d.contains(e.target)) d.open = false; });
  });

  $("jobs").addEventListener("click", (e) => {
    const btn = e.target.closest("[data-do]");
    if (!btn) return;
    const id = btn.closest("[data-id]").dataset.id;
    const j = S.jobs.find((x) => x.id === id) || { id, ...(MY.mine[id] || {}).job };
    const act = btn.dataset.do;
    if (act === "saved" || act === "applied") {
      const cur = MY.mine[id];
      if (cur && cur.status === act) delete MY.mine[id];
      else MY.mine[id] = { status: act, at: new Date().toISOString(),
        job: { title: j.title, company: j.company, url: j.url, location: j.location } };
    } else if (act === "hide-job") MY.hidden.add(id);
    else if (act === "hide-co") MY.hiddenCo.add(coKey(j.company));
    saveMine();
    render();
  });

  // Filters that differ from the defaults, as removable chips. Returns how many are on.
  const AGE_DEFAULT = "45";
  function renderActive(f, count) {
    const chips = [];
    if (f.q) chips.push(["q", `“${f.q}”`]);
    document.querySelectorAll("#families input:not(:checked)").forEach((i) => chips.push([`fam:${i.value}`, `No ${S.labels[i.value]}`]));
    if (f.level) chips.push(["level", LEVELS[f.level]]);
    if (f.state) chips.push(["state", f.state]);
    if (String(f.age) !== AGE_DEFAULT) chips.push(["age", `Posted within ${f.age} days`]);
    if (f.remote) chips.push(["remote", "Remote only"]);
    if (f.h1b) chips.push(["h1b", "H-1B sponsors only"]);
    if (f.visaOk) chips.push(["visaOk", "Open to visa holders"]);
    if (f.entry) chips.push(["entry", "Entry level / new grad"]);
    if (f.newOnly) chips.push(["newOnly", "New since last visit"]);
    const hidden = S.jobs.filter(isHidden).length;
    if (hidden) chips.push(["hidden", `${hidden} hidden — show them`]);
    $("active").hidden = !chips.length;
    $("active").innerHTML = chips.length ? `<span class="muted">${count.toLocaleString()} jobs ·</span>
      ${chips.map(([k, label]) => `<button type="button" class="fchip" data-filter="${esc(k)}" title="Remove this filter">${esc(label)} <span aria-hidden="true">✕</span><span class="sr">remove</span></button>`).join("")}
      <button type="button" class="link small" data-act="clear-filters">Clear all</button>` : "";
    return chips.length;
  }

  function clearFilter(k) {
    if (k === "q") $("q").value = "";
    else if (k.startsWith("fam:")) document.querySelector(`#families input[value="${k.slice(4)}"]`).checked = true;
    else if (k === "level" || k === "state") $(k).value = "";
    else if (k === "age") $("age").value = AGE_DEFAULT;
    else if (k === "hidden") { MY.hidden.clear(); MY.hiddenCo.clear(); saveMine(); }
    else $(k).checked = false;
  }

  document.addEventListener("click", (e) => {
    const chip = e.target.closest("[data-filter]");
    const all = e.target.closest('[data-act="clear-filters"]');
    if (!chip && !all) return;
    if (chip) clearFilter(chip.dataset.filter);
    else {
      ["q", "level", "state", "age", "remote", "h1b", "visaOk", "entry", "newOnly"].forEach((k) => clearFilter(k));
      document.querySelectorAll("#families input").forEach((i) => { i.checked = true; });
    }
    S.limit = PAGE;
    render();
  });

  function check(kind, text, title) {
    const [mark, said] = { ok: ["✓", "Meets"], warn: ["~", "Close"], no: ["✗", "Short"] }[kind];
    return `<span class="ck ${kind}" title="${esc(title)}"><span class="mark" aria-hidden="true">${mark}</span>` +
      `<span class="sr">${said}: </span>${esc(text)}</span>`;
  }

  init().catch((e) => { $("meta").textContent = "Couldn't load jobs.json. Run export.py first."; console.error(e); });
})();
