// UI: load jobs, read the résumé locally, filter, render.
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const PAGE = 40;
  const LEVELS = { entry: "Entry", mid: "Mid", senior: "Senior", manager: "Manager / Lead", director: "Director+" };
  const S = { ctx: null, jobs: [], labels: {}, profile: null, scores: null, tab: "all", limit: PAGE, text: "" };

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
    $("pasteBtn").addEventListener("click", (e) => { e.preventDefault(); $("paste").hidden = false; $("paste").focus(); });
    $("paste").addEventListener("input", () => { S.text = $("paste").value; $("matchBtn").disabled = S.text.trim().length < 200; });
    $("matchBtn").addEventListener("click", () => runMatch(true));
    $("years").addEventListener("change", () => S.profile && runMatch(false));
    $("clearBtn").addEventListener("click", clearResume);
  }

  async function readFile(f) {
    status(`Reading ${f.name}…`);
    try {
      const name = f.name.toLowerCase();
      if (name.endsWith(".pdf")) S.text = await pdfText(await f.arrayBuffer());
      else if (name.endsWith(".docx")) S.text = (await mammoth.extractRawText({ arrayBuffer: await f.arrayBuffer() })).value;
      else S.text = await f.text();
      if (S.text.trim().length < 200) return status("Couldn't find much text in that file. If it's a scanned PDF, paste the text instead.", true);
      $("paste").value = S.text;
      runMatch(true);
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

  function runMatch(fresh) {
    S.profile = Match.analyzeResume(S.ctx, S.text);
    if (fresh || $("years").value === "") $("years").value = S.profile.years;
    S.profile.years = +$("years").value || 0;
    S.scores = Match.scoreAll(S.ctx, S.profile);
    S.tab = "strong"; S.limit = PAGE;
    $("matchBtn").disabled = false; $("clearBtn").hidden = false;
    status("");
    renderProfile();
    render();
    $("profile").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function clearResume() {
    Object.assign(S, { profile: null, scores: null, text: "", tab: "all", limit: PAGE });
    $("paste").value = ""; $("paste").hidden = true; $("file").value = ""; $("years").value = "";
    $("matchBtn").disabled = true; $("clearBtn").hidden = true;
    $("profile").hidden = true; $("tips").hidden = true;
    render();
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

  // ---------- rendering ----------
  function renderProfile() {
    const p = S.profile, labels = S.labels;
    const fam = Object.entries(p.affinity).sort((a, b) => b[1] - a[1]).filter(([, v]) => v >= 0.35).map(([k]) => labels[k]);
    const terms = [...p.terms].map((id) => S.ctx.byId[id].label);
    $("profile").hidden = false;
    $("profile").innerHTML = `
      <div><h2>What we read from your résumé</h2>
        <p class="muted">Leaning <strong>${esc(fam.join(", ") || "general quality")}</strong> · ${p.years} years experience (edit above if wrong) · ${terms.length} RA/QA terms found</p></div>
      <div class="chips">${terms.map((t) => `<span class="chip ok">${esc(t)}</span>`).join("")}</div>`;
  }

  function renderTips(list, f) {
    const t = Match.tips(S.ctx, S.profile, S.scores, list, S.jobs, f, S.labels);
    $("tips").hidden = !t.length;
    $("tips").innerHTML = `<h2>Ways to widen your search</h2><div class="tip-grid">${t.map((x) => `
      <div class="tip tip-${x.kind}"><h3>${esc(x.title)}</h3><p>${esc(x.body)}</p>
      ${x.chips ? `<div class="chips">${x.chips.map((c) => `<span class="chip gap" title="in ${c.n} of ${c.of} close matches">${esc(c.label)} <b>${c.n}</b></span>`).join("")}</div>` : ""}
      </div>`).join("")}</div>`;
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
    const visa = j.h1b_ra_qa > 0 ? `<span class="badge visa" title="H-1B LCAs for RA/QA roles, DOL FY2024–26">H-1B: ${j.h1b_ra_qa} RA/QA filings</span>`
      : j.cap_exempt ? `<span class="badge visa soft" title="Universities and hospitals can usually sponsor cap-exempt">May be cap-exempt</span>`
      : j.h1b_all > 0 ? `<span class="badge soft" title="H-1B LCAs in any role, DOL FY2024–26">H-1B: ${j.h1b_all} filings (other roles)</span>` : "";
    const chips = s ? `<div class="chips small">
        ${s.matched.slice(0, 5).map((id) => `<span class="chip ok">${esc(S.ctx.byId[id].label)}</span>`).join("")}
        ${s.missing.slice(0, 3).map((id) => `<span class="chip gap" title="Asked for, not on your résumé">${esc(S.ctx.byId[id].label)}</span>`).join("")}
      </div>` : "";
    return `<li class="job card">
      ${s ? `<div class="score ${s.score >= Match.STRONG ? "hi" : s.score >= Match.WEAK ? "mid" : "lo"}" title="Match score">${s.score}</div>` : ""}
      <div class="job-main">
        <a class="job-title" href="${esc(j.url)}" target="_blank" rel="noopener nofollow">${esc(j.title)}</a>
        <div class="job-sub"><strong>${esc(j.company)}</strong> · ${esc(j.location || "United States")}${j.remote ? ' · <span class="remote">Remote</span>' : ""}</div>
        <div class="badges"><span class="badge fam">${esc(S.labels[j.family])}</span>
          <span class="badge">${LEVELS[j.seniority]}${j.years != null ? ` · ${j.years}+ yrs` : ""}</span>
          <span class="badge soft">${age === 0 ? "today" : age + "d ago"}</span>${visa}</div>
        ${chips}
      </div>
      <div class="job-actions">
        <a class="btn small" href="${esc(j.url)}" target="_blank" rel="noopener nofollow">Apply on employer site</a>
        <a class="link small" href="${li}" target="_blank" rel="noopener nofollow">Find the hiring team on LinkedIn</a>
      </div></li>`;
  }

  init().catch((e) => { $("meta").textContent = "Couldn't load jobs.json. Run export.py first."; console.error(e); });
})();
