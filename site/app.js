// UI: load jobs, read the résumé locally, filter, render.
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const PAGE = 40, TIPS_SHOWN = 3;
  const LEVELS = { entry: "Entry", mid: "Mid", senior: "Senior", manager: "Manager / Lead", director: "Director+" };
  const US = /^(?:us|usa|united states(?: of america)?)$/i;
  const GENERIC = new Set(["fda", "audit", "sop"]); // nearly every posting has these; not worth a chip
  const S = { ctx: null, jobs: [], labels: {}, profile: null, scores: null, tab: "all", limit: PAGE,
    text: "", years: null, showSkills: false, showAllTips: false };

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const daysAgo = (d) => Math.max(0, Math.round((Date.now() - new Date(d + "T12:00:00")) / 864e5));

  async function init() {
    const [vocab, data] = await Promise.all([
      fetch("vocab.json").then((r) => r.json()),
      fetch("jobs.json", { cache: "no-cache" }).then((r) => r.json()),
    ]);
    S.jobs = data.jobs;
    S.ctx = Match.compile(vocab, S.jobs);
    S.labels = Object.fromEntries(vocab.families.map((f) => [f.id, f.label]));
    $("meta").textContent = `${data.count.toLocaleString()} open roles · updated ${new Date(data.generated).toLocaleDateString(undefined, { month: "short", day: "numeric" })}`;

    $("families").innerHTML = vocab.families.map((f) =>
      `<label class="check"><input type="checkbox" value="${f.id}" checked> ${esc(f.label)} <span class="count">${S.jobs.filter((j) => j.family === f.id).length}</span></label>`).join("");
    $("level").insertAdjacentHTML("beforeend", Object.entries(LEVELS).map(([k, v]) => `<option value="${k}">${v}</option>`).join(""));
    const states = [...new Set(S.jobs.map((j) => j.state).filter(Boolean))].sort();
    $("state").insertAdjacentHTML("beforeend", states.map((s) => `<option>${s}</option>`).join(""));

    document.querySelectorAll(".filters input, .filters select").forEach((el) =>
      el.addEventListener(el.type === "search" ? "input" : "change", () => { S.limit = PAGE; render(); }));
    wireUpload();
    render();
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
    S.scores = Match.scoreAll(S.ctx, S.profile);
    S.tab = "strong"; S.limit = PAGE;
    document.body.classList.add("matched");
    status("");
    renderProfile();
    render();
    if (first) $("profile").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function clearResume() {
    Object.assign(S, { profile: null, scores: null, text: "", years: null, tab: "all", limit: PAGE, showSkills: false, showAllTips: false });
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
      remote: $("remote").checked, h1b: $("h1b").checked,
    };
  }

  function applyFilters(f) {
    return S.jobs.filter((j) =>
      f.families.has(j.family) && (!f.level || j.seniority === f.level) &&
      (!f.state || j.state === f.state || (j.remote && !j.state)) && (!f.remote || j.remote) &&
      (!f.h1b || Match.hasSponsorTrack(j)) && daysAgo(j.posted) <= f.age &&
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
    const current = tabs.find((t) => t[0] === S.tab) || tabs[0];
    $("tabs").innerHTML = tabs.map(([id, label, l]) =>
      `<button role="tab" aria-selected="${id === current[0]}" data-tab="${id}">${label} <span class="count">${l.length}</span></button>`).join("");
    $("tabs").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => { S.tab = b.dataset.tab; S.limit = PAGE; render(); }));

    const rows = current[2];
    $("jobs").innerHTML = rows.length ? rows.slice(0, S.limit).map(card).join("")
      : `<li class="empty">No jobs match these filters. Try widening them.</li>`;
    $("more").hidden = rows.length <= S.limit;
    $("more").onclick = () => { S.limit += PAGE; render(); };
  }

  function card(j) {
    const s = S.scores && S.scores.get(j.id);
    const age = daysAgo(j.posted);
    const role = j.family === "RA" ? "regulatory affairs" : j.family === "QC" ? "quality control" : "quality";
    const li = "https://www.linkedin.com/search/results/people/?keywords=" + encodeURIComponent(`${j.company} ${role} manager`);
    const visa = j.h1b_ra_qa > 0 ? `<span class="badge visa" title="${j.h1b_ra_qa} H-1B applications for RA/QA roles, DOL FY2024–26">H-1B sponsor</span>`
      : j.cap_exempt ? `<span class="badge visa soft" title="Universities and hospitals can usually sponsor H-1B cap-exempt">May be cap-exempt</span>` : "";
    const meta = [LEVELS[j.seniority] + (j.years != null ? ` · ${j.years}+ yrs` : ""), age === 0 ? "today" : `${age}d ago`].join(" · ");
    const total = s ? s.matched.length + s.missing.length : 0;
    const fit = s && total ? `<div class="fit"><span class="muted">${s.matched.length} of ${total} skills</span>
        ${s.missing.filter((id) => !GENERIC.has(id)).slice(0, 3).map((id) => `<span class="chip gap" title="Asked for, not on your résumé">${esc(S.ctx.byId[id].label)}</span>`).join("")}</div>` : "";
    return `<li class="job card">
      ${s ? `<div class="score ${s.score >= Match.STRONG ? "hi" : s.score >= Match.WEAK ? "mid" : "lo"}" title="Match score">${s.score}</div>` : ""}
      <div class="job-main">
        <a class="job-title" href="${esc(j.url)}" target="_blank" rel="noopener nofollow">${esc(j.title)}</a>
        <div class="job-sub"><strong>${esc(j.company)}</strong> · ${esc(shortLoc(j))}${j.remote && !/remote/i.test(shortLoc(j)) ? ' · <span class="remote">Remote</span>' : ""} ${visa}</div>
        <div class="job-meta">${meta} · <a href="${li}" target="_blank" rel="noopener nofollow" title="LinkedIn people search">Find hiring team</a></div>
        ${fit}
      </div></li>`;
  }

  init().catch((e) => { $("meta").textContent = "Couldn't load jobs.json. Run export.py first."; console.error(e); });
})();
