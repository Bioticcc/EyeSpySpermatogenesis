"use strict";
// Annotation page. Every label is POSTed (and committed) immediately; failed
// saves wait in an outbox and are retried, so nothing is lost if the server
// hiccups.

const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "text") n.textContent = v;
    else if (k === "class") n.className = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
  for (const k of kids) n.append(k);
  return n;
};
const store = {
  get(k) { try { return JSON.parse(sessionStorage.getItem(k)); } catch { return null; } },
  set(k, v) { try { sessionStorage.setItem(k, JSON.stringify(v)); } catch { /* ignore */ } },
  del(k) { try { sessionStorage.removeItem(k); } catch { /* ignore */ } },
};

const S = {
  cfg: null, tubules: [], byId: new Map(), labels: new Map(), session: null,
  queue: [], qi: -1, current: null, fixing: false, outline: true,
  view: { s: 1, x: 0, y: 0 }, outbox: [], lastSaved: null,
  run: null, preds: new Map(), paperStage: new Map(), blind: false,
};

// ---------- API ----------
async function api(path, body) {
  const opt = body === undefined ? {} : {
    method: "POST", body: JSON.stringify(body),
    headers: { "Content-Type": "application/json", "X-Token": window.TOKEN },
  };
  const r = await fetch(path, opt);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) { const e = new Error(data.error || r.statusText); e.status = r.status; throw e; }
  return data;
}

function toast(msg, bad = false, ms = 4500) {
  const t = $("toast");
  t.textContent = msg; t.className = "toast" + (bad ? " bad" : ""); t.hidden = false;
  clearTimeout(toast.h); toast.h = setTimeout(() => (t.hidden = true), ms);
}

function setSaveState() {
  const s = $("save-state");
  if (S.outbox.length) {
    s.className = "save-state bad";
    s.textContent = `⚠ ${S.outbox.length} label(s) not saved yet, retrying…`;
  } else {
    s.className = "save-state";
    s.textContent = S.lastSaved ? `✓ All saved · ${S.lastSaved}` : "";
  }
}

async function flushOutbox() {
  while (S.outbox.length) {
    const job = S.outbox[0];
    try {
      await api(job.path, job.body);
      S.outbox.shift();
      S.lastSaved = new Date().toLocaleTimeString();
    } catch (e) {
      if (e.status === 401) { S.outbox = []; toast("Your session ended. Please log in again.", true); showLogin(); break; }
      if (e.status && e.status >= 400 && e.status < 500) { S.outbox.shift(); toast("Not saved: " + e.message, true); continue; }
      break; // network/server problem: keep and retry later
    }
  }
  setSaveState();
}
setInterval(flushOutbox, 5000);

function send(path, body) {
  S.outbox.push({ path, body });
  setSaveState();
  flushOutbox();
}

// ---------- data ----------
async function loadLabels() {
  if (!S.session) return;
  const rows = await api("/api/labels?mode=" + S.session.mode);
  // keep optimistic local values for jobs still in the outbox
  const pending = new Set(S.outbox.map((j) => j.body.tubule_id));
  const next = new Map();
  for (const r of rows) {
    if (pending.has(r.tubule_id)) continue;
    if (!next.has(r.tubule_id)) next.set(r.tubule_id, []);
    next.get(r.tubule_id).push(r);
  }
  for (const tid of pending) next.set(tid, S.labels.get(tid) || []);
  S.labels = next;
}
setInterval(async () => {
  try { await loadLabels(); if (!$("view-annotate").hidden) { renderSide(); if (seg()) repaintSeg(); } } catch { /* retry later */ }
}, 20000);

const labelsOf = (tid) => S.labels.get(tid) || [];
const mineOf = (tid) => labelsOf(tid).find((l) => l.origin === "web" && l.annotator === S.session.name);
const paperOf = (tid) => labelsOf(tid).find((l) => l.origin === "paper");
const othersOf = (tid) => labelsOf(tid).filter((l) => !(l.origin === "web" && l.annotator === S.session.name));
const staging = () => S.session && S.session.mode === "staging";
const review = () => S.session && S.session.mode === "model_review";
const seg = () => S.session && S.session.mode === "segmentation";
const choices = () => (staging() || review() ? S.cfg.stages : S.cfg.verdicts);
const nameOf = (v) => (choices().find((c) => c[0] === v) || [v, v ?? "—"])[1];

function localSet(tid, origin, annotator, value, detail) {
  const arr = labelsOf(tid).filter((l) => !(l.origin === origin && l.annotator === annotator));
  if (value !== undefined) {
    const prev = labelsOf(tid).find((l) => l.origin === origin && l.annotator === annotator);
    arr.push({ tubule_id: tid, mode: S.session.mode, annotator, origin, value,
      detail: { ...(prev ? prev.detail : {}), ...(detail || {}) } });
  }
  S.labels.set(tid, arr);
}

async function loadReview() {
  S.run = (S.cfg.review_runs || [])[0] || null;
  S.preds = new Map();
  S.paperStage = new Map();
  S.blind = !!store.get("blind");
  if (!S.run) { toast("No model review has been set up yet.", true, 8000); return; }
  for (const p of await api("/api/predictions?run=" + encodeURIComponent(S.run))) {
    const r = S.preds.get(p.tubule_id) || { ord: p.ord };
    r[p.slot] = { model: p.model, value: p.value };
    S.preds.set(p.tubule_id, r);
  }
  for (const l of await api("/api/labels?mode=staging")) if (l.origin === "paper" && l.value) S.paperStage.set(l.tubule_id, l.value);
}

// ---------- queue ----------
const QUEUES = {
  staging: [
    ["mine", "Not staged by me"],
    ["unlabelled", "No stage from anyone yet"],
    ["paper_check", "Paper readings to check"],
    ["all", "All accepted tubules"],
  ],
  model_review: [
    ["mine", "Not answered by me"],
    ["all", "All tubules in this review"],
  ],
  segmentation: [
    ["mine", "Not reviewed by me"],
    ["rec", "Accepted only"],
    ["flagged", "Flagged only"],
    ["all", "All candidates"],
  ],
};

function inPool(t) {
  if (review()) return S.preds.has(t.tubule_id);
  return staging() ? t.recommended === 1 : true;
}

function matches(t) {
  const q = $("f-queue").value, src = $("f-source").value;
  if (!inPool(t) || (src !== "all" && t.source_folder !== src)) return false;
  const labs = labelsOf(t.tubule_id);
  if (q === "mine") return !mineOf(t.tubule_id);
  if (q === "unlabelled") return !labs.some((l) => l.value);
  if (q === "paper_check") { const p = paperOf(t.tubule_id); return !!(p && p.detail.needs_check); }
  if (q === "rec") return t.recommended === 1;
  if (q === "flagged") return t.recommended === 0;
  return true;
}

function buildQueue(keepCurrent = false) {
  const cur = S.current;
  const list = S.tubules.filter(matches);
  if (review()) {
    // review order is fixed (shuffled when the review run was made)
    const done = (t) => (mineOf(t.tubule_id) ? 1 : 0);
    const ord = (t) => S.preds.get(t.tubule_id).ord;
    list.sort((a, b) => ($("f-queue").value === "mine" ? done(a) - done(b) : 0) || ord(a) - ord(b));
  } else if ($("f-queue").value === "mine") {
    // coverage first: tubules nobody has labelled yet, then the rest
    const done = (t) => (labelsOf(t.tubule_id).some((l) => l.value) ? 1 : 0);
    list.sort((a, b) => done(a) - done(b) || a.seq - b.seq);
  }
  S.queue = list.map((t) => t.tubule_id);
  S.qi = keepCurrent && cur ? S.queue.indexOf(cur) : -1;
  store.set("filters", { source: $("f-source").value, queue: $("f-queue").value });
  if (seg() && !SEG.single) { renderSegGrid(); if (S.queue.length) segSelect(Math.max(0, S.qi), false); return; }
  if (!keepCurrent || S.qi < 0) advance(1);
  else show(cur);
}

function advance(dir) {
  // forward: next queue entry that still matches; back: simply the previous one
  if (dir < 0) {
    if (S.qi > 0) { S.qi -= 1; show(S.queue[S.qi]); }
    return;
  }
  for (let i = S.qi + 1; i < S.queue.length; i++) {
    if (matches(S.byId.get(S.queue[i]))) { S.qi = i; return show(S.queue[i]); }
  }
  for (let i = 0; i <= S.qi && i < S.queue.length; i++) {
    if (matches(S.byId.get(S.queue[i])) && S.queue[i] !== S.current) { S.qi = i; return show(S.queue[i]); }
  }
  S.qi = S.queue.length;
  show(null);
}

// ---------- rendering ----------
function show(tid) {
  S.current = tid; S.fixing = false;
  resetView();
  $("empty").hidden = !!tid;
  if (!tid) {
    $("img").removeAttribute("src"); $("outline").removeAttribute("src");
    $("empty").textContent = "Nothing left in this list. Choose another section or list, or log out.";
    $("position").textContent = "";
    renderSide();
    return;
  }
  const t = S.byId.get(tid);
  $("img").src = "/ws/" + t.image;
  $("outline").src = "/ws/" + t.outline;
  preload();
  const remaining = S.queue.filter((q) => matches(S.byId.get(q))).length;
  $("position").textContent = `${S.qi + 1} / ${S.queue.length} · ${remaining} left in this list`;
  renderSide();
}

function preload() {
  for (let i = S.qi + 1, n = 0; i < S.queue.length && n < 3; i++, n++) {
    const t = S.byId.get(S.queue[i]);
    new Image().src = "/ws/" + t.image;
  }
}

function renderSide() {
  const tid = S.current;
  const t = tid && S.byId.get(tid);
  $("meta-title").textContent = t ? `Tubule #${t.label}` : "Tubule";
  $("meta").innerHTML = "";
  $("others").innerHTML = "";
  $("paper").hidden = true;
  $("paper-fixing").hidden = true;
  $("review").hidden = !review() || !t;
  if (!t) { $("mine").textContent = "—"; markKeys(null); return; }
  $("meta").append(
    el("div", {}, el("b", { text: t.source_name })),
    el("div", { text: t.recommended ? "Accepted crop" : "Flagged: " + (t.flags || "—").replaceAll(";", ", ") }),
  );
  if (t.display_scale < 1) $("meta").append(el("div", { text: `Shown at ${Math.round(t.display_scale * 100)}% (large crop)` }));
  const mine = mineOf(tid);
  $("mine").textContent = mine ? nameOf(mine.value) : "—";
  const others = review() && !mine ? [] : othersOf(tid).filter((l) => l.origin === "web");
  if (review() && !mine) $("others").append(el("div", { class: "none", text: "Other reviewers' answers appear after yours." }));
  else if (!others.length) $("others").append(el("div", { class: "none", text: "No one else has labelled this yet." }));
  for (const o of others) {
    const cls = !mine ? "other" : o.value === mine.value ? "other agree" : "other disagree";
    $("others").append(el("div", { class: cls }, el("span", { text: o.annotator }), el("b", { text: nameOf(o.value) })));
  }
  if (review()) renderReview(tid, mine);
  const p = staging() && paperOf(tid);
  if (p) {
    $("paper").hidden = false;
    $("paper").classList.toggle("needs-check", !!p.detail.needs_check);
    $("paper-who").textContent = "· " + p.annotator;
    const snips = $("paper-snips");
    snips.innerHTML = "";
    for (const sn of p.detail.snippets || []) snips.append(el("img", { src: "/ws/" + sn, alt: "Handwritten mark" }));
    const reads = (p.detail.stage_reads || []).map((r) => (r ? nameOf(r) : "?"));
    $("paper-read").textContent = p.value ? nameOf(p.value) : reads.length > 1 ? reads.join(" + ") : "Unreadable";
    const notes = [];
    if (p.detail.transcription_checked_by) notes.push(`Reading checked by ${p.detail.transcription_checked_by}`);
    const reader = /^Claude/.test(p.detail.transcribed_by || "") ? "Claude's reading of the photo" : "Machine reading";
    if (!p.detail.transcription_checked_by && p.detail.needs_check) notes.push(reader + ", please check: " + (p.detail.read_flags || []).map(flagText).join("; "));
    else if (!p.detail.transcription_checked_by) notes.push(reader + " of the handwriting (not yet checked by a person)");
    $("paper-flags").textContent = notes.join(" · ");
    $("paper-flags").style.color = p.detail.needs_check ? "" : "var(--muted)";
    $("paper-ok").disabled = !p.value;
  }
  markKeys(mine ? mine.value : null);
}

function renderReview(tid, mine) {
  const pr = S.preds.get(tid) || {};
  $("review-run").textContent = "· " + (S.run || "");
  const hide = S.blind && !mine;
  for (const [id, v] of [["rv-paper", S.paperStage.get(tid)], ["rv-a", (pr.A || {}).value], ["rv-b", (pr.B || {}).value]]) {
    const e = $(id);
    e.textContent = hide ? "hidden" : v ? nameOf(v) : "—";
    e.className = hide ? "hid" : mine && v ? (v === mine.value ? "ok" : "no") : "";
  }
  $("rv-blind").checked = S.blind;
}

const FLAG_TEXT = {
  uncircled: "numeral had no circle",
  marks_disagree_or_unreadable: "this crop has marks with different stages (likely two tubules)",
  several_marks_same_stage: "this crop has 2+ marks with the same stage",
  bar_under_x_not_found: "XI or XII unsure",
  "3_bars_under_x": "XI or XII unsure",
  x_borderline_height: "X unsure",
  wide_ink_maybe_note_attached: "extra ink beside the numeral",
  annotator_note_nearby_eg_question_mark: "note (e.g. ?) beside the mark",
  unreadable_needs_transcription: "could not be read",
  extra_red_ink_beside_numeral_check_meaning: "extra red ink beside the numeral (dot, symbol or crossed-out note)",
  not_a_stage_numeral: "the mark is a symbol (e.g. germ-cell depletion, transition region), not a stage",
  annotator_note_nearby: "note beside the mark",
  uncircled_mark_transferred_needs_check: "numeral had no circle",
};
function flagText(f) {
  if (FLAG_TEXT[f]) return FLAG_TEXT[f];
  if (/^bar_peaks_/.test(f)) return "I or II–III unsure";
  return f.replaceAll("_", " ");
}

function markKeys(value) {
  for (const b of $("keys").querySelectorAll(".keybtn")) b.classList.toggle("selected", b.dataset.value === value);
}

function buildKeys() {
  const k = $("keys");
  k.innerHTML = "";
  k.className = staging() || review() ? "keys" : "keys verdicts";
  for (const [value, label, key] of choices()) {
    const wide = value === "cant_stage";
    const keyText = value === "II-III" ? "2 / 3" : key === "b" ? "B" : key;
    k.append(el("button", { class: "keybtn" + (wide ? " wide" : ""), "data-value": value, title: `Key ${keyText}`,
      onclick: () => choose(value) }, el("strong", { text: label }), el("kbd", { text: keyText })));
  }
}

// ---------- actions ----------
function choose(value) {
  const tid = S.current;
  if (!tid) return;
  if (S.fixing) return fixTranscription(value);
  localSet(tid, "web", S.session.name, value, review() ? { run: S.run } : undefined);
  send("/api/label", { session_id: S.session.id, tubule_id: tid, value, ...(review() ? { run: S.run } : {}) });
  renderSide();
  if (!(review() && S.blind)) setTimeout(() => advance(1), review() ? 600 : 140);
}

function clearMine() {
  const tid = S.current;
  if (!tid || !mineOf(tid)) return;
  localSet(tid, "web", S.session.name, undefined);
  send("/api/label", { session_id: S.session.id, tubule_id: tid, value: null });
  renderSide();
}

function confirmTranscription() {
  const tid = S.current, p = tid && paperOf(tid);
  if (!p || !p.value) return;
  fixTranscription(p.value);
}

function fixTranscription(value) {
  const tid = S.current, p = paperOf(tid);
  S.fixing = false;
  if (!p || value === "cant_stage") { renderSide(); return; }
  localSet(tid, "paper", p.annotator, value, { transcription_checked_by: S.session.name, needs_check: false });
  send("/api/transcription", { session_id: S.session.id, tubule_id: tid, value });
  renderSide();
  toast(`Paper reading saved as ${nameOf(value)}. Now give your own stage.`, false, 2500);
}

// ---------- zoom / pan ----------
function applyView() {
  const v = S.view;
  $("stage-img").style.transform = `translate(${v.x}px, ${v.y}px) scale(${v.s})`;
}
function resetView() { S.view = { s: 1, x: 0, y: 0 }; applyView(); }
(() => {
  const vw = $("viewer");
  vw.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = vw.getBoundingClientRect(), v = S.view;
    const f = Math.exp(-e.deltaY * 0.0015), s = Math.min(8, Math.max(1, v.s * f));
    const px = e.clientX - r.left, py = e.clientY - r.top;
    v.x = px - ((px - v.x) * s) / v.s; v.y = py - ((py - v.y) * s) / v.s; v.s = s;
    if (s === 1) { v.x = 0; v.y = 0; }
    applyView();
  }, { passive: false });
  let drag = null;
  vw.addEventListener("mousedown", (e) => { drag = { x: e.clientX - S.view.x, y: e.clientY - S.view.y }; vw.classList.add("dragging"); });
  window.addEventListener("mousemove", (e) => { if (drag) { S.view.x = e.clientX - drag.x; S.view.y = e.clientY - drag.y; applyView(); } });
  window.addEventListener("mouseup", () => { drag = null; vw.classList.remove("dragging"); });
  vw.addEventListener("dblclick", resetView);
})();

function setOutline(on) {
  S.outline = on;
  $("viewer").classList.toggle("no-outline", !on);
}

// ---------- keyboard ----------
document.addEventListener("keydown", (e) => {
  if (!S.session || $("view-annotate").hidden) return;
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (seg() && !SEG.single) { segKey(e, k); return; }
  if (seg() && k === "escape") { e.preventDefault(); backToGrid(); return; }
  if (k === "arrowright" || k === " ") { e.preventDefault(); advance(1); return; }
  if (k === "arrowleft") { e.preventDefault(); advance(-1); return; }
  if (k === "o") { setOutline(!S.outline); return; }
  if (k === "escape") { S.fixing = false; renderSide(); return; }
  if (k === "delete" || k === "backspace") { e.preventDefault(); clearMine(); return; }
  if (staging() && k === "t") { confirmTranscription(); return; }
  if (staging() && k === "f" && paperOf(S.current)) { S.fixing = true; $("paper-fixing").hidden = false; return; }
  let hit = choices().find((c) => c[2] === k);
  if ((staging() || review()) && k === "3") hit = choices().find((c) => c[0] === "II-III");
  if (hit) { e.preventDefault(); choose(hit[0]); }
});

// ---------- views ----------
function setView(name) {
  for (const v of ["login", "annotate", "overview", "progress", "help"]) $("view-" + v).hidden = v !== name;
  for (const b of document.querySelectorAll(".nav-item[data-view]")) b.classList.toggle("active", b.dataset.view === name);
  if (name === "progress") renderProgress();
  if (name === "overview") renderOverviewTab();
  if (name === "help") renderHelp();
}

function showLogin() {
  S.session = null; store.del("session");
  $("nav").hidden = true; $("who").hidden = true;
  $("recent-names").innerHTML = "";
  for (const n of (S.cfg.annotators || []).filter((n) => n !== "paper-import").slice(0, 12)) {
    $("recent-names").append(el("button", { class: "chip", text: n, onclick: () => { $("name").value = n; } }));
  }
  setView("login");
  $("name").focus();
}

async function startSession(sess) {
  S.session = sess; store.set("session", sess);
  $("nav").hidden = false; $("who").hidden = false;
  $("who").innerHTML = "";
  $("who").append(el("span", { text: sess.name }), el("small", { text: staging() ? "Staging mode" : review() ? "Model review" : "Segmentation review" }));
  $("nav-overview").hidden = !review();
  SEG.single = false;
  $("seg-wrap").hidden = !seg();
  $("work").hidden = seg();
  $("seg-back").hidden = true;
  $("prev").hidden = seg(); $("next").hidden = seg();
  if (review()) await loadReview();
  buildKeys();
  const fs = $("f-source");
  fs.innerHTML = "";
  fs.append(el("option", { value: "all", text: "All sections" }));
  const seen = new Set();
  for (const t of S.tubules) if (!seen.has(t.source_folder)) {
    seen.add(t.source_folder);
    fs.append(el("option", { value: t.source_folder, text: t.source_name.replace(/\.tiff?$/i, "") }));
  }
  const fq = $("f-queue");
  fq.innerHTML = "";
  for (const [v, l] of QUEUES[sess.mode]) fq.append(el("option", { value: v, text: l }));
  const f = store.get("filters");
  if (f) { if ([...fs.options].some((o) => o.value === f.source)) fs.value = f.source; if ([...fq.options].some((o) => o.value === f.queue)) fq.value = f.queue; }
  await loadLabels();
  // Model review opens on Overview, staging on Progress (owner, 2026-11-11).
  setView(review() ? "overview" : staging() ? "progress" : "annotate");
  buildQueue();
}

async function login() {
  const name = $("name").value.trim();
  const mode = document.querySelector("input[name=mode]:checked").value;
  $("login-error").hidden = true;
  if (!name) { $("login-error").textContent = "Please type your name."; $("login-error").hidden = false; return; }
  try {
    const r = await api("/api/login", { name, mode });
    await startSession({ id: r.session_id, name: r.name, mode: r.mode });
  } catch (e) { $("login-error").textContent = e.message; $("login-error").hidden = false; }
}

async function logout() {
  await flushOutbox();
  if (S.outbox.length && !confirm("Some labels are not saved yet. Log out anyway?")) return;
  try {
    const r = await api("/api/logout", { session_id: S.session.id });
    toast(`Logged out. ${r.labels_this_session} label(s) this session. CSV saved: ${r.export}`, false, 9000);
  } catch (e) { toast("Logout export failed: " + e.message, true, 9000); }
  S.cfg = await api("/api/config");
  $("name").value = "";
  showLogin();
}

// ---------- progress ----------
function renderProgress() {
  if (review()) return renderReviewProgress();
  $("stage-panel").hidden = false;
  $("by-source").closest(".panel").querySelector("h2").textContent = "By section";
  const st = staging();
  const pool = S.tubules.filter(inPool);
  $("progress-title").textContent = st ? "Staging progress" : "Segmentation review progress";
  $("progress-lede").textContent = st
    ? "Accepted (unflagged) tubules only. Paper stages are counted separately from on-screen labels."
    : "All candidates, accepted and flagged.";
  const web = (tid) => labelsOf(tid).filter((l) => l.origin === "web" && l.value);
  const anyStage = (tid) => labelsOf(tid).some((l) => l.value);
  const n = pool.length, done1 = pool.filter((t) => anyStage(t.tubule_id)).length;
  const done2 = pool.filter((t) => new Set(web(t.tubule_id).map((l) => l.annotator)).size >= 2).length;
  const papers = pool.filter((t) => paperOf(t.tubule_id)).length;
  const check = pool.filter((t) => { const p = paperOf(t.tubule_id); return p && p.detail.needs_check; }).length;
  const mine = pool.filter((t) => mineOf(t.tubule_id)).length;
  const m = $("metrics");
  m.innerHTML = "";
  const card = (num, lab) => m.append(el("div", { class: "metric-card" }, el("strong", { text: String(num) }), el("span", { text: lab })));
  card(n, st ? "accepted tubules" : "candidates");
  card(`${done1} (${n ? Math.round((100 * done1) / n) : 0}%)`, st ? "have a stage" : "reviewed");
  card(done2, "labelled by 2+ people");
  if (st) { card(papers, "paper stages"); card(check, "paper readings to check"); }
  card(mine, "labelled by you");

  const bySrc = new Map();
  for (const t of pool) {
    if (!bySrc.has(t.source_folder)) bySrc.set(t.source_folder, { name: t.source_name, n: 0, any: 0, two: 0, paper: 0, check: 0, mine: 0 });
    const r = bySrc.get(t.source_folder), p = paperOf(t.tubule_id);
    r.n++; r.any += anyStage(t.tubule_id); r.two += new Set(web(t.tubule_id).map((l) => l.annotator)).size >= 2;
    r.paper += !!p; r.check += !!(p && p.detail.needs_check); r.mine += !!mineOf(t.tubule_id);
  }
  const tb = $("by-source");
  tb.innerHTML = "";
  const head = st ? ["Section", "Accepted", "Have a stage", "2+ people", "Paper", "To check", "You"] : ["Section", "Candidates", "Reviewed", "2+ people", "You"];
  tb.append(el("tr", {}, ...head.map((h) => el("th", { text: h }))));
  for (const [folder, r] of bySrc) {
    const cells = st ? [r.name, r.n, r.any, r.two, r.paper, r.check, r.mine] : [r.name, r.n, r.any, r.two, r.mine];
    const tr = el("tr", st ? { class: "clickable", title: "Open the section map" } : {}, ...cells.map((c) => el("td", { text: String(c) })));
    if (st) tr.addEventListener("click", () => openSectionMap(r.name, folder).catch((e) => toast("Section map: " + e.message, true)));
    tb.append(tr);
  }

  const bars = $("stage-bars");
  bars.innerHTML = "";
  $("stage-panel").querySelector("h2").textContent = st ? "Labels per stage" : "Labels per verdict";
  const counts = new Map(choices().map((c) => [c[0], { web: 0, paper: 0 }]));
  for (const t of pool) for (const l of labelsOf(t.tubule_id)) if (l.value && counts.has(l.value)) counts.get(l.value)[l.origin === "paper" ? "paper" : "web"]++;
  const max = Math.max(1, ...[...counts.values()].map((c) => c.web + c.paper));
  bars.append(el("div", { class: "legend" }, el("i", { style: "background:var(--green-500)" }), "on screen", el("i", { style: "background:var(--blue)" }), "paper"));
  for (const [v, c] of counts) {
    const track = el("div", { class: "bar-track" },
      el("div", { class: "bar", style: `width:${(100 * c.web) / max}%` }),
      el("div", { class: "bar paper", style: `width:${(100 * c.paper) / max}%` }));
    bars.append(el("div", { class: "bar-row" }, el("span", { text: nameOf(v).replace("Can't stage / bad crop", "Can't stage") }), track, el("span", { text: String(c.web + c.paper) })));
  }

  const people = new Map();
  for (const t of pool) for (const l of labelsOf(t.tubule_id)) {
    const key = l.annotator + (l.origin === "paper" ? " (paper)" : "");
    const r = people.get(key) || { n: 0, last: "" };
    r.n += l.value ? 1 : 0; r.last = l.updated_at > r.last ? l.updated_at : r.last;
    people.set(key, r);
  }
  const tp = $("by-person");
  tp.innerHTML = "";
  tp.append(el("tr", {}, el("th", { text: "Annotator" }), el("th", { text: "Labels" }), el("th", { text: "Last label" })));
  for (const [k, r] of [...people].sort((a, b) => b[1].n - a[1].n)) {
    tp.append(el("tr", {}, el("td", { text: k }), el("td", { text: String(r.n) }), el("td", { text: (r.last || "").replace("T", " ").slice(0, 16) })));
  }
  $("export-note").textContent = S.cfg.last_backup ? `Last automatic backup: ${S.cfg.last_backup.replace("T", " ").slice(0, 16)}` : "";
}

function renderReviewProgress() {
  const stg = S.cfg.stages.map((c) => c[0]);
  const cyc = (a, b) => { const d = Math.abs(stg.indexOf(a) - stg.indexOf(b)) % 11; return Math.min(d, 11 - d); };
  $("progress-title").textContent = "Model review";
  $("progress-lede").textContent = `Review run ${S.run || "—"}. Scores compare each model and the paper stage with the reviewer's own stage ("can't stage" answers are left out).`;
  const ids = [...S.preds.keys()];
  const answers = new Map(); // annotator -> [{tid, value}]
  for (const tid of ids) for (const l of labelsOf(tid)) if (l.origin === "web" && l.value) {
    if (!answers.has(l.annotator)) answers.set(l.annotator, []);
    answers.get(l.annotator).push({ tid, value: l.value });
  }
  const m = $("metrics");
  m.innerHTML = "";
  const card = (num, lab) => m.append(el("div", { class: "metric-card" }, el("strong", { text: String(num) }), el("span", { text: lab })));
  card(ids.length, "tubules in this review");
  card(ids.filter((t) => mineOf(t)).length, "answered by you");
  card(answers.size, "reviewers");
  // scores per reviewer and model
  const models = [...new Set(ids.flatMap((t) => ["A", "B"].map((s) => (S.preds.get(t)[s] || {}).model)).filter(Boolean))].sort();
  const tb = $("by-source");
  tb.innerHTML = "";
  tb.closest(".panel").querySelector("h2").textContent = "Scores against each reviewer";
  tb.append(el("tr", {}, ...["Reviewer", "Answered", "Compared with", "Exact", "Within one stage"].map((h) => el("th", { text: h }))));
  for (const [who, list] of answers) {
    const staged = list.filter((x) => x.value !== "cant_stage");
    const rowsFor = (name, get) => {
      const pairs = staged.map((x) => [get(x.tid), x.value]).filter(([p]) => p);
      const ex = pairs.filter(([p, v]) => p === v).length, w1 = pairs.filter(([p, v]) => cyc(p, v) <= 1).length;
      const pct = (n) => (pairs.length ? `${Math.round((100 * n) / pairs.length)}% (${n}/${pairs.length})` : "—");
      tb.append(el("tr", {}, ...[who, String(list.length), name, pct(ex), pct(w1)].map((c) => el("td", { text: c }))));
    };
    for (const model of models) rowsFor(model, (t) => { const p = S.preds.get(t); return ["A", "B"].map((s) => p[s]).find((x) => x && x.model === model)?.value; });
    rowsFor("Paper stage (Yeongseok)", (t) => S.paperStage.get(t));
  }
  if (!answers.size) tb.append(el("tr", {}, el("td", { colspan: "5", text: "No answers yet." })));
  $("stage-panel").hidden = true;
  const tp = $("by-person");
  tp.innerHTML = "";
  tp.append(el("tr", {}, el("th", { text: "Reviewer" }), el("th", { text: "Answers" })));
  for (const [who, list] of answers) tp.append(el("tr", {}, el("td", { text: who }), el("td", { text: String(list.length) })));
  $("export-note").textContent = S.cfg.last_backup ? `Last automatic backup: ${S.cfg.last_backup.replace("T", " ").slice(0, 16)}` : "";
}

// ---------- model review overview ----------
const OV = { data: null };
const pct = (x) => (x === null || x === undefined ? "—" : `${Math.round(100 * x)}%`);

async function loadOverviews() {
  if (OV.data) return OV.data;
  const out = [];
  for (const run of S.cfg.overview_runs || []) {
    try {
      const r = await fetch(`/ws/review/${encodeURIComponent(run)}/overview.json`, { cache: "no-store" });
      if (r.ok) out.push(await r.json());
    } catch { /* run without overlays */ }
  }
  OV.data = out; // run-name order: round 1 (Baseline), round 2 (SATINN), ...
  return OV.data;
}

async function renderOverview() {
  const runs = await loadOverviews();
  const sel = $("ov-section");
  if (!runs.length) { $("ov-grid").innerHTML = ""; $("ov-grid").append(el("p", { text: "No overview has been generated yet." })); return; }
  const sections = [...new Set(runs.flatMap((r) => r.sections.map((s) => s.section)))].sort();
  if (sel.options.length !== sections.length) {
    const keep = sel.value || store.get("ov-section");
    sel.innerHTML = "";
    for (const s of sections) sel.append(el("option", { value: s, text: s }));
    if (sections.includes(keep)) sel.value = keep;
  }
  store.set("ov-section", sel.value);
  const show = $("ov-show").value, sec = sel.value;
  // legend
  const lg = $("ov-legend");
  lg.innerHTML = "";
  const cols = show === "agree" ? runs[0].agree_colours : runs[0].colours;
  for (const [k, c] of Object.entries(cols)) lg.append(el("span", {}, el("i", { style: `background:${c}` }), k.replace("II-III", "II–III")));
  if (show !== "agree") lg.append(el("span", {}, el("i", { style: "background:#969696" }), "grey / uncoloured: no stage, flagged crop, or reading still to check (not evaluated)"));
  // one row per method: paper reference first, then that method's models.
  // Same column count in every row so panels line up across rows.
  const grid = $("ov-grid");
  grid.innerHTML = "";
  const ncols = Math.max(...runs.map((r) => 1 + r.models.length));
  for (const r of runs) {
    const s = r.sections.find((x) => x.section === sec);
    if (!s) continue;
    const cards = el("div", { class: "ov-cards", style: `--cols:${ncols}` });
    cards.append(panelCard("Paper stage (Yeongseok)", "", `/ws/review/${r.run}/${s.paper}`, "Reference: unchecked transcription of the hand-staged sheet",
      sec, () => methodCtx(runs, sec, r.run, null)));
    for (const [name, m] of Object.entries(s.models)) {
      cards.append(panelCard(name, "", `/ws/review/${r.run}/${show === "agree" ? m.agree : m.stage}`,
        `This section: exact ${pct(m.exact)}, within one ${pct(m.within1)} (n=${m.n})`, sec, () => methodCtx(runs, sec, r.run, name)));
    }
    grid.append(el("section", { class: "ov-row" }, el("h2", { class: "ov-row-title", text: r.method }), cards));
  }
  // statistics table
  const t = $("ov-stats");
  t.innerHTML = "";
  t.append(el("tr", {}, ...["Method", "Model", "Section n", "Section exact", "Section within one", "All exact", "All balanced", "All within one", "All n"].map((h) => el("th", { text: h }))));
  for (const r of runs) {
    const s = r.sections.find((x) => x.section === sec);
    for (const mo of r.models) {
      const m = s ? s.models[mo.name] : null, o = mo.overall;
      t.append(el("tr", {}, ...[r.method, mo.name, m ? m.n : "—", pct(m && m.exact), pct(m && m.within1),
        pct(o.acc), pct(o.balanced_acc), pct(o.within1), o.n].map((c) => el("td", { text: String(c) }))));
    }
  }
  // per-stage recall
  const stg = S.cfg.stages.map((c) => c[0]).filter((v) => v !== "cant_stage");
  const tr = $("ov-recall");
  tr.innerHTML = "";
  tr.append(el("tr", {}, el("th", { text: "Model" }), ...stg.map((s) => el("th", { text: s.replace("II-III", "II–III") }))));
  for (const r of runs) for (const mo of r.models) {
    tr.append(el("tr", {}, el("td", { text: mo.name }), ...stg.map((s) => el("td", { text: pct(mo.recall[s]) }))));
  }
}

// ---------- overview tabs: Method overview | EyeSpy (owner, 2026-11-13) ----------
function renderOverviewTab(tab) {
  tab = tab || store.get("ov-tab") || "methods";
  if (!["methods", "eyespy"].includes(tab)) tab = "methods";
  store.set("ov-tab", tab);
  for (const t of ["methods", "eyespy"]) {
    $("ov-pane-" + t).hidden = t !== tab;
    $("ov-tab-" + t).classList.toggle("active", t === tab);
    $("ov-tab-" + t).setAttribute("aria-selected", String(t === tab));
  }
  const run = tab === "eyespy" ? renderEyeSpy : renderOverview;
  run().catch((e) => toast("Overview: " + e.message, true));
}

const ES = { data: null };
const ROLE = { primary: "primary", "settings variant": "settings", comparison: "comparison", experiment: "experiment" };

async function loadEyeSpy() {
  if (ES.data) return ES.data;
  const r = await fetch("/ws/review/eyespy/index.json", { cache: "no-store" });
  ES.data = r.ok ? await r.json() : null;
  return ES.data;
}

function smoothingText(s) {
  return s.on ? `on (k=${s.k}, λ=${s.lam})` : "off";
}

async function renderEyeSpy() {
  const d = await loadEyeSpy();
  const grid = $("es-grid");
  grid.innerHTML = "";
  if (!d || !d.models.length) {
    grid.append(el("p", { text: "No EyeSpy model has been saved yet (python -m eyespy_spermatogenesis.staging.eyespy train)." }));
    return;
  }
  $("es-strategy").textContent = d.strategy + " SATINN and UNI2 stay in the Method overview tab as reference models.";
  const show = $("es-show").value;
  const models = d.models.filter((m) => ($("es-comparison").checked || m.role !== "comparison") && ($("es-experiments").checked || m.role !== "experiment"));
  // legend
  const lg = $("es-legend");
  lg.innerHTML = "";
  for (const [k, c] of Object.entries(show === "agree" ? d.agree_colours : d.colours)) {
    lg.append(el("span", {}, el("i", { style: `background:${c}` }), k.replace("II-III", "II–III")));
  }
  // one row per sample section: ground truth, then every saved model
  const base = "/ws/review/eyespy/";
  for (const s of d.samples) {
    const cards = el("div", { class: "ov-cards", style: `--cols:${1 + models.length}` });
    cards.append(panelCard("Ground truth", s.truth_label.replace(/^Ground truth: /, ""), base + s.truth, "Reference labels; unchecked by a second person",
      s.section, async () => eyespyCtx(s, models, null)));
    for (const m of models) {
      const p = m.panels[s.section];
      if (!p) continue;
      cards.append(panelCard(m.name, `trained ${m.created.slice(0, 10)} on ${m.training.n} tubules`, base + (show === "agree" ? p.agree : p.stage),
        `This section: exact ${pct(p.exact)}, within one ${pct(p.within1)} (n=${p.n})`, s.section, async () => eyespyCtx(s, models, m)));
    }
    const stages = s.stages.map((x) => x.replace("II-III", "II–III")).join(", ");
    grid.append(el("section", { class: "ov-row" }, el("h2", { class: "ov-row-title", text: `${s.section}  ·  stages ${stages}` }), cards));
  }
  // history table
  const sgn = (x) => `${x >= 0 ? "+" : "−"}${Math.abs(100 * x).toFixed(1)}`;
  const ci = (c) => (c ? `${sgn(c[0])} to ${sgn(c[1])}` : "—");
  const t = $("es-history");
  t.innerHTML = "";
  t.append(el("tr", {}, ...["Model", "Training labels", "Smoothing", "Class weighting", "Exact", "Balanced", "Within one",
    "Paper sections exact", "Paper sections balanced", "Δ exact vs parent", "Δ exact vs previous version"].map((h) => el("th", { text: h }))));
  for (const m of models) {
    const src = m.training.sources.map((x) => `${x.annotator} ${x.origin} (${x.n_evaluated})`).join(" + ");
    const nameCell = el("td", {}, m.name, el("span", { class: "es-role " + (ROLE[m.role] || ""), text: ROLE[m.role] || m.role }));
    const prev = m.vs_previous_version;
    t.append(el("tr", {}, nameCell, ...[src, smoothingText(m.settings.smoothing), m.settings.class_weighting ? "on" : "off",
      pct(m.all.acc), pct(m.all.balanced_acc), pct(m.all.within1),
      pct(m.paper_sections && m.paper_sections.acc), pct(m.paper_sections && m.paper_sections.balanced_acc),
      ci(m.vs_parent && m.vs_parent.delta_exact_ci95), prev ? `${ci(prev.delta_exact_ci95)} vs ${prev.against.replace(/ \(.*\)$/, "")}` : "—"].map((c) => el("td", { text: String(c) }))));
  }
  // per-stage recall
  const stg = S.cfg.stages.map((c) => c[0]).filter((v) => v !== "cant_stage");
  const tr = $("es-recall");
  tr.innerHTML = "";
  tr.append(el("tr", {}, el("th", { text: "Model" }), el("th", { text: "n" }), ...stg.map((s) => el("th", { text: s.replace("II-III", "II–III") }))));
  for (const m of models) {
    tr.append(el("tr", {}, el("td", { text: m.name }), el("td", { text: String(m.all.n) }), ...stg.map((s) => el("td", { text: pct(m.all.recall[s]) }))));
  }
  // settings
  const st = $("es-settings");
  st.innerHTML = "";
  st.append(el("tr", {}, ...["Model", "Saved", "Checkpoint", "Classifier strength (alpha) per member", "Training stages", "Code"].map((h) => el("th", { text: h }))));
  for (const m of models) {
    const alpha = Object.entries(m.settings.alpha_checkpoint).map(([k, v]) => `${k.split(":")[0]}: ${v}`).join("; ");
    const per = Object.entries(m.training.per_stage).filter(([, n]) => n).map(([k, n]) => `${k} ${n}`).join(", ");
    st.append(el("tr", {}, ...[m.name, m.created.slice(0, 16).replace("T", " "), m.checkpoint, alpha, per, m.code_commit].map((c) => el("td", { text: String(c) }))));
  }
}

// ---------- segmentation mode: grid of crops, as in the original review sheet (owner, 2026-11-13) ----------
// Click a crop or use ← →, press 1–7; the next crop is selected. 0 / Delete clears. Double-click: full size.
const SEG = { shown: 0, page: 240, single: false, obs: null };
function segCard(i) {
  const tid = S.queue[i], t = S.byId.get(tid);
  const thumb = el("div", { class: "thumb" },
    el("img", { src: "/ws/" + t.image, alt: `Crop ${t.label}`, loading: "lazy" }),
    el("img", { class: "ol", src: "/ws/" + t.outline, alt: "", loading: "lazy" }),
    el("span", { class: "badge" }));
  const where = $("f-source").value === "all" ? t.source_name.replace(/\.tiff?$/i, "") : "";
  const card = el("div", { class: "tg-card" + (t.recommended ? "" : " flagged"), tabindex: "0", "data-i": String(i),
    title: t.recommended ? "Accepted crop" : "Flagged: " + (t.flags || "—").replaceAll(";", ", ") },
    thumb, el("div", { class: "meta" }, el("span", { text: `#${t.label}` }), el("span", { text: where })));
  card.addEventListener("click", () => segSelect(i, false));
  card.addEventListener("dblclick", () => openSingle(i));
  paintSeg(card);
  return card;
}
function paintSeg(card) {
  const tid = S.queue[Number(card.dataset.i)], m = mineOf(tid);
  if (m && m.value) { card.dataset.v = m.value; card.querySelector(".badge").textContent = nameOf(m.value); }
  else { delete card.dataset.v; }
}
function repaintSeg() { for (const c of $("seg-grid").querySelectorAll(".tg-card")) paintSeg(c); segCount(); }
function segCount() {
  const done = S.queue.filter((tid) => mineOf(tid)).length;
  $("seg-count").textContent = `${S.queue.length} crops in this list · ${done} reviewed by you`;
}
function renderSegGrid() {
  const grid = $("seg-grid");
  grid.innerHTML = "";
  SEG.shown = 0;
  if (!S.queue.length) grid.append(el("p", { class: "meta", text: "Nothing in this list. Choose another section or list." }));
  segMore();
  segCount();
}
function segMore() {
  const grid = $("seg-grid");
  const old = grid.querySelector(".more");
  if (old) old.remove();
  const end = Math.min(S.queue.length, SEG.shown + SEG.page);
  for (let i = SEG.shown; i < end; i++) grid.append(segCard(i));
  SEG.shown = end;
  if (end < S.queue.length) {
    const more = el("button", { class: "button secondary small more", text: `Show more (${S.queue.length - end} left)`, onclick: segMore });
    grid.append(more);
    if (SEG.obs) SEG.obs.disconnect();
    SEG.obs = new IntersectionObserver((es) => { if (es.some((x) => x.isIntersecting)) segMore(); }, { root: grid, rootMargin: "400px" });
    SEG.obs.observe(more);
  }
}
function segSelect(i, scroll = true) {
  if (i < 0 || i >= S.queue.length) return;
  while (i >= SEG.shown) segMore();
  const grid = $("seg-grid");
  const prev = grid.querySelector(".tg-card.selected");
  if (prev) prev.classList.remove("selected");
  const card = grid.querySelector(`.tg-card[data-i="${i}"]`);
  S.qi = i; S.current = S.queue[i];
  if (card) { card.classList.add("selected"); if (scroll) card.scrollIntoView({ block: "nearest" }); }
  $("position").textContent = `${i + 1} / ${S.queue.length}`;
}
function segSet(value) {
  const tid = S.current;
  if (!tid || S.qi < 0) return;
  localSet(tid, "web", S.session.name, value === null ? undefined : value);
  send("/api/label", { session_id: S.session.id, tubule_id: tid, value });
  const card = $("seg-grid").querySelector(`.tg-card[data-i="${S.qi}"]`);
  if (card) paintSeg(card);
  segCount();
  if (value !== null) segSelect(S.qi + 1);
}
function segKey(e, k) {
  if (k === "arrowright") { e.preventDefault(); segSelect(S.qi + 1); return; }
  if (k === "arrowleft") { e.preventDefault(); segSelect(S.qi - 1); return; }
  if (k === "o") { $("seg-outline").checked = !$("seg-outline").checked; segOutline(); return; }
  if (k === "0" || k === "delete" || k === "backspace") { e.preventDefault(); segSet(null); return; }
  if (k === "enter") { e.preventDefault(); openSingle(S.qi); return; }
  const hit = choices().find((c) => c[2] === k);
  if (hit) { e.preventDefault(); segSet(hit[0]); }
}
const segOutline = () => $("seg-grid").classList.toggle("no-outline", !$("seg-outline").checked);
// Full size: the normal single-crop view; Esc or "Back to grid" returns.
function openSingle(i) {
  if (i < 0 || i >= S.queue.length) return;
  SEG.single = true;
  $("seg-wrap").hidden = true; $("work").hidden = false; $("seg-back").hidden = false;
  $("prev").hidden = false; $("next").hidden = false;
  S.qi = i; show(S.queue[i]);
}
function backToGrid() {
  SEG.single = false;
  $("seg-wrap").hidden = false; $("work").hidden = true; $("seg-back").hidden = true;
  $("prev").hidden = true; $("next").hidden = true;
  const i = Math.max(0, S.qi);
  renderSegGrid();
  segSelect(Math.min(i, S.queue.length - 1));
}
function bindSegGrid() {
  $("seg-outline").addEventListener("change", segOutline);
  $("seg-size").addEventListener("input", () => { $("seg-grid").style.setProperty("--thumb", $("seg-size").value + "px"); store.set("seg-size", $("seg-size").value); });
  const sz = store.get("seg-size");
  if (sz) { $("seg-size").value = sz; $("seg-grid").style.setProperty("--thumb", sz + "px"); }
  $("seg-back").addEventListener("click", backToGrid);
  const keys = $("seg-keys");
  for (const [value, label, key] of S.cfg.verdicts) {
    keys.append(el("button", { class: "keybtn", title: `Key ${key}`, onclick: () => segSet(value) }, el("strong", { text: label }), el("kbd", { text: key })));
  }
}

// ---------- staging Progress: whole-section map coloured by the stages so far (owner, 2026-11-13) ----------
// Click a tubule to open it in Annotate. Images: review/sections/<section>.jpg + review/maps/<section>.png.
function stageRGB(k) {
  const n = STG().length, h = k / n, sat = 0.8, v = k % 2 === 0 ? 0.95 : 0.72;
  const i = Math.floor(h * 6), f = h * 6 - i, p = v * (1 - sat), q = v * (1 - sat * f), t = v * (1 - sat * (1 - f));
  const [r, g, b] = [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]][i % 6];
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}
const CANT_RGB = [150, 150, 150], FLAG_RGB = [70, 70, 70];
const SM = { section: null, sourceName: null, map: null, ov: null };
const OVS = new Map();
function overviewPixels(section) {
  if (!OVS.has(section)) {
    OVS.set(section, (async () => {
      const r = await fetch(`/ws/review/sections/${encodeURIComponent(section.replace(/^#/, ""))}.jpg`);
      if (!r.ok) throw new Error("no overview image for this section (run clickmaps sections)");
      const bmp = await createImageBitmap(await r.blob());
      const c = document.createElement("canvas");
      c.width = bmp.width; c.height = bmp.height;
      const g = c.getContext("2d", { willReadFrequently: true });
      g.drawImage(bmp, 0, 0);
      return { w: bmp.width, h: bmp.height, d: g.getImageData(0, 0, bmp.width, bmp.height).data };
    })());
    OVS.get(section).catch(() => OVS.delete(section));
  }
  return OVS.get(section);
}
function smLabel(tid) {
  const mine = mineOf(tid);
  if ($("sm-who").value === "mine") return mine && mine.value ? { value: mine.value, who: "you" } : null;
  if (mine && mine.value) return { value: mine.value, who: "you" };
  const web = labelsOf(tid).filter((l) => l.origin === "web" && l.value).sort((a, b) => (b.updated_at || "").localeCompare(a.updated_at || ""));
  if (web.length) return { value: web[0].value, who: web[0].annotator };
  const p = paperOf(tid);
  return p && p.value ? { value: p.value, who: p.annotator + ", paper" } : null;
}
async function openSectionMap(sourceName, folder) {
  const section = sourceName.replace(/\.tiff?$/i, "");
  Object.assign(SM, { section, sourceName, folder });
  $("sm-title").textContent = section;
  $("sm-summary").textContent = "Loading…";
  $("sm").hidden = false;
  const [map, ov] = await Promise.all([hitMap(section), overviewPixels(section)]);
  if (map.w !== ov.w || map.h !== ov.h) throw new Error("overview and click map sizes differ");
  SM.map = map; SM.ov = ov;
  drawSectionMap();
  $("sm-close").focus();
}
function drawSectionMap() {
  const { w, h, d: md } = SM.map, od = SM.ov.d;
  const tubs = S.tubules.filter((t) => t.source_name === SM.sourceName);
  const maxLab = Math.max(0, ...tubs.map((t) => t.label));
  const col = new Int16Array((maxLab + 1) * 3).fill(-1), text = new Map();
  const stg = STG();
  let n = 0, staged = 0, mineN = 0;
  for (const t of tubs) {
    const o = t.label * 3;
    if (!t.recommended) { col.set(FLAG_RGB, o); continue; }
    n++;
    if (mineOf(t.tubule_id)) mineN++;
    const lab = smLabel(t.tubule_id);
    if (!lab) continue;
    staged++;
    if (lab.value === "cant_stage") { col.set(CANT_RGB, o); continue; }
    const k = stg.indexOf(lab.value);
    if (k >= 0) { col.set(stageRGB(k), o); text.set(t.label, lab.value.replace("II-III", "II")); }
  }
  const cv = $("sm-canvas");
  cv.width = w; cv.height = h;
  const g = cv.getContext("2d");
  const img = g.createImageData(w, h), d = img.data;
  const sx = new Float64Array(maxLab + 1), sy = new Float64Array(maxLab + 1), cnt = new Float64Array(maxLab + 1);
  const L = (i) => md[i * 4] + 256 * md[i * 4 + 1];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x, j = i * 4, lab = L(i);
      const grey = (od[j] + od[j + 1] + od[j + 2]) / 3;
      let r = 0.35 * od[j] + 0.65 * grey, gg = 0.35 * od[j + 1] + 0.65 * grey, b = 0.35 * od[j + 2] + 0.65 * grey;
      if (lab && lab <= maxLab) {
        sx[lab] += x; sy[lab] += y; cnt[lab]++;
        const o = lab * 3;
        if (col[o] >= 0) { r = 0.45 * r + 0.55 * col[o]; gg = 0.45 * gg + 0.55 * col[o + 1]; b = 0.45 * b + 0.55 * col[o + 2]; }
      }
      const right = x + 1 < w ? L(i + 1) : lab, down = y + 1 < h ? L(i + w) : lab;
      if ((right !== lab || down !== lab) && (lab || right || down)) { r = 255; gg = 255; b = 255; }
      d[j] = r; d[j + 1] = gg; d[j + 2] = b; d[j + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0);
  g.font = `bold ${Math.max(11, Math.round(w / 55))}px sans-serif`;
  g.textAlign = "center"; g.textBaseline = "middle"; g.lineWidth = Math.max(3, Math.round(w / 300)); g.strokeStyle = "#fff"; g.fillStyle = "#000";
  for (const [lab, tx] of text) {
    if (!cnt[lab]) continue;
    const x = sx[lab] / cnt[lab], y = sy[lab] / cnt[lab];
    g.strokeText(tx, x, y); g.fillText(tx, x, y);
  }
  const wrap = cv.parentElement;
  cv.style.height = Math.min(wrap.clientHeight, (wrap.clientWidth * h) / w) + "px";
  cv.style.width = "auto";
  $("sm-summary").textContent = `${n} accepted tubules: ${staged} have a stage (${pct(n ? staged / n : null)})`
    + `${$("sm-who").value === "mine" ? " from you" : ""}, ${mineN} labelled by you. Uncoloured tubules have no stage yet.`;
  const lg = $("sm-legend");
  lg.innerHTML = "";
  stg.forEach((s, k) => lg.append(el("span", {}, el("i", { style: `background:rgb(${stageRGB(k)})` }), stageName(s))));
  lg.append(el("span", {}, el("i", { style: `background:rgb(${CANT_RGB})` }), "Can't stage"),
    el("span", {}, el("i", { style: `background:rgb(${FLAG_RGB})` }), "Flagged crop (not staged)"),
    el("span", {}, el("i", { style: "background:#d8d8d8" }), "No stage yet"));
}
async function smTubuleAt(e) {
  const lab = await labelAt(SM.section, $("sm-canvas"), e);
  return lab ? tubuleOf(SM.section, lab) : null;
}
function goAnnotate(t) {
  $("sm").hidden = true;
  setView("annotate");
  $("f-source").value = t.source_folder;
  let note = "";
  if (!matches(t)) { $("f-queue").value = "all"; note = " The list is now “All accepted tubules” because this one was not in your list."; }
  S.current = t.tubule_id;
  buildQueue(true);
  toast(`Tubule #${t.label} of ${SM.section}.${note}`, false, 3500);
}
function bindSectionMap() {
  $("sm-close").addEventListener("click", () => { $("sm").hidden = true; });
  $("sm").addEventListener("click", (e) => { if (e.target === $("sm")) $("sm").hidden = true; });
  $("sm-who").addEventListener("change", () => SM.map && drawSectionMap());
  const cv = $("sm-canvas");
  cv.addEventListener("click", async (e) => {
    const t = await smTubuleAt(e);
    if (!t) { toast("Click inside a tubule."); return; }
    if (!t.recommended) { toast(`Tubule #${t.label} is a flagged crop and is not staged.`); return; }
    goAnnotate(t);
  });
  let busy = false;
  cv.addEventListener("mousemove", async (e) => {
    if (busy) return;
    busy = true;
    try {
      const t = await smTubuleAt(e);
      const lab = t && smLabel(t.tubule_id);
      $("sm-hover").textContent = !t ? "Click a tubule to label it in Annotate."
        : !t.recommended ? `#${t.label} · flagged crop (not staged)`
        : `#${t.label} · ${lab ? nameOf(lab.value) + " (" + lab.who + ")" : "no stage yet"} · click to label`;
    } finally { busy = false; }
  });
  document.addEventListener("keydown", (e) => { if (!$("sm").hidden && e.key === "Escape") { e.preventDefault(); $("sm").hidden = true; } });
}

// ---------- click a tubule on a panel: grid of that stage, then one tubule (owner, 2026-11-13) ----------
// Read-only. Click maps: review/maps/<section>.png (candidate label = red + 256 * green).
const STG = () => S.cfg.stages.map((c) => c[0]).filter((v) => v !== "cant_stage");
const stageName = (s) => (s || "—").replace("II-III", "II–III");
function cycDist(a, b) {
  const st = STG(), k = st.length, d = Math.abs(st.indexOf(a) - st.indexOf(b)) % k;
  return Math.min(d, k - d);
}
const scoreOf = (p, t) => (!p || !t ? "none" : ["exact", "one"][cycDist(p, t)] || "far");
const SCORE_TEXT = { exact: "correct", one: "1 stage off", far: "2+ stages off", none: "no ground truth" };
const SCORE_COLOUR = { exact: "var(--green-700)", one: "var(--amber)", far: "var(--red)", none: "var(--muted)" };

const MAPS = new Map();
function hitMap(section) {
  if (!MAPS.has(section)) {
    MAPS.set(section, (async () => {
      const r = await fetch(`/ws/review/maps/${encodeURIComponent(section.replace(/^#/, ""))}.png`);
      if (!r.ok) throw new Error("no click map for this section yet");
      const bmp = await createImageBitmap(await r.blob(), { colorSpaceConversion: "none", premultiplyAlpha: "none" });
      const c = document.createElement("canvas");
      c.width = bmp.width; c.height = bmp.height;
      const g = c.getContext("2d", { willReadFrequently: true });
      g.drawImage(bmp, 0, 0);
      return { w: bmp.width, h: bmp.height, d: g.getImageData(0, 0, bmp.width, bmp.height).data };
    })());
    MAPS.get(section).catch(() => MAPS.delete(section));
  }
  return MAPS.get(section);
}
async function labelAt(section, img, e) {
  const m = await hitMap(section), r = img.getBoundingClientRect();
  const x = Math.floor(((e.clientX - r.left) / r.width) * m.w), y = Math.floor(((e.clientY - r.top) / r.height) * m.h);
  if (x < 0 || y < 0 || x >= m.w || y >= m.h) return 0;
  const i = (y * m.w + x) * 4;
  return m.d[i] + 256 * m.d[i + 1];
}
let TUB_BY_SEC = null;
function tubuleOf(section, label) {
  if (!TUB_BY_SEC) {
    TUB_BY_SEC = new Map();
    for (const t of S.tubules) TUB_BY_SEC.set(t.source_name.replace(/\.tiff?$/i, "") + "|" + t.label, t);
  }
  return TUB_BY_SEC.get(section + "|" + label);
}

// One panel card; clicking a tubule opens the grid for its stage in this panel.
function panelCard(title, sub, src, stat, section, ctxFn) {
  const img = el("img", { src, alt: title, loading: "lazy", title: "Click a tubule to see every tubule of its stage" });
  img.addEventListener("click", async (e) => {
    try {
      const [ctx, label] = await Promise.all([ctxFn(), labelAt(section, img, e)]);
      if (!label) { toast("Click inside a tubule."); return; }
      const st = ctx.stages[label];
      if (!st) { toast(`Tubule #${label} has no stage in this panel (not labelled, flagged crop, or not scored).`); return; }
      openGrid(ctx, st, label);
    } catch (err) { toast("Tubules: " + err.message, true); }
  });
  const full = el("a", { class: "full", href: src, target: "_blank", rel: "noopener", title: "Open this panel full size", text: "⤢" });
  return el("div", { class: "ov-card" }, el("h3", {}, title, full, sub ? el("div", { class: "es-sub", text: sub }) : ""), img, el("div", { class: "stat", text: stat }));
}

// Method overview: stages.json per run and section (all runs share the paper truth).
const ST_CACHE = new Map();
async function runStages(run, section) {
  const key = run + "|" + section;
  if (!ST_CACHE.has(key)) {
    ST_CACHE.set(key, fetch(`/ws/review/${encodeURIComponent(run)}/${encodeURIComponent(section.replace(/^#/, ""))}/stages.json`, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null)).catch(() => null));
  }
  return ST_CACHE.get(key);
}
async function methodCtx(runs, section, run, name) {
  const all = await Promise.all(runs.map(async (r) => ({ r, st: await runStages(r.run, section) })));
  const mine = all.find((a) => a.r.run === run).st;
  if (!mine) throw new Error("no per-tubule stages for this run (run clickmaps backfill)");
  const peers = all.filter((a) => a.st).flatMap((a) => Object.entries(a.st.models).map(([n, stg]) => ({ title: n, group: a.r.method, stages: stg })));
  const truthName = "Paper stage (Yeongseok)";
  const group = runs.find((r) => r.run === run).method;
  return { section, title: name || truthName, group, isTruth: !name, stages: name ? mine.models[name] || {} : mine.truth, truth: mine.truth, truthName, peers };
}
function eyespyCtx(s, models, m) {
  if (!s.truth_stages) throw new Error("no per-tubule stages yet (rebuild the EyeSpy index)");
  const truthName = "Ground truth: " + s.truth_label.replace(/^Ground truth: /, "");
  const peers = models.filter((x) => x.panels[s.section] && x.panels[s.section].stages).map((x) => ({ title: x.name, group: "EyeSpy", stages: x.panels[s.section].stages }));
  return { section: s.section, title: m ? m.name : truthName, group: "EyeSpy", isTruth: !m, stages: m ? m.panels[s.section].stages || {} : s.truth_stages, truth: s.truth_stages, truthName, peers };
}

// ---- grid
const TG = { ctx: null, list: [], focus: null };
function openGrid(ctx, stage, focus) {
  TG.ctx = ctx; TG.focus = focus;
  const present = new Set([...Object.values(ctx.stages), ...Object.values(ctx.truth)]);
  const sel = $("tg-stage");
  sel.innerHTML = "";
  for (const s of STG()) if (present.has(s)) sel.append(el("option", { value: s, text: stageName(s) }));
  sel.value = stage;
  $("tg-by").value = ctx.isTruth ? "truth" : "panel";
  $("tg-by").disabled = ctx.isTruth;
  $("tg-filter").value = "all";
  $("tg-filter").disabled = ctx.isTruth;
  $("tg").hidden = false;
  renderGrid();
  const f = document.querySelector("#tg-grid .tg-card.focus");
  (f || $("tg-close")).focus();
}
function renderGrid() {
  const c = TG.ctx, st = $("tg-stage").value, by = $("tg-by").value, filt = $("tg-filter").value;
  const src = by === "truth" ? c.truth : c.stages;
  const rows = Object.keys(src).filter((k) => src[k] === st).map(Number).map((l) => ({
    l, t: tubuleOf(c.section, l), p: c.stages[l], g: c.truth[l], s: c.isTruth ? "none" : scoreOf(c.stages[l], c.truth[l]),
  })).filter((r) => r.t);
  const n = { exact: 0, one: 0, far: 0, none: 0 };
  for (const r of rows) n[r.s]++;
  const order = { far: 0, one: 1, exact: 2, none: 3 };
  TG.list = rows.filter((r) => filt === "all" || r.s === filt)
    .sort((a, b) => (b.l === TG.focus) - (a.l === TG.focus) || order[a.s] - order[b.s] || a.l - b.l);
  $("tg-eyebrow").textContent = `${c.section} · ${c.title}`;
  $("tg-title").textContent = c.isTruth ? `Stage ${stageName(st)}: ground truth`
    : by === "truth" ? `Ground-truth stage ${stageName(st)}: what this model called them` : `Stage ${stageName(st)} according to this model`;
  const scored = n.exact + n.one + n.far;
  $("tg-summary").textContent = c.isTruth ? `${rows.length} tubules (${c.truthName}).`
    : `${rows.length} tubules. Against ${c.truthName.replace(/^Ground truth: /, "the ground truth, ")}: ${n.exact} correct (${pct(scored ? n.exact / scored : null)}), `
      + `${n.one} one stage off, ${n.far} two or more off${n.none ? `, ${n.none} without ground truth` : ""}. Wrong ones first.`;
  const grid = $("tg-grid");
  grid.innerHTML = "";
  grid.classList.toggle("no-outline", !$("tg-outline").checked);
  TG.list.forEach((r, i) => {
    const thumb = el("div", { class: "thumb" },
      el("img", { src: "/ws/" + r.t.image, alt: `Tubule ${r.l}`, loading: "lazy" }),
      el("img", { class: "ol", src: "/ws/" + r.t.outline, alt: "", loading: "lazy" }));
    if (!c.isTruth) thumb.append(el("span", { class: "badge", text: SCORE_TEXT[r.s] }));
    const meta = el("div", { class: "meta" }, el("span", { text: `#${r.l}` }),
      el("span", {}, c.isTruth ? "" : "Model ", el("b", { text: stageName(c.isTruth ? r.g : r.p) }), c.isTruth ? "" : " · truth ", c.isTruth ? "" : el("b", { text: stageName(r.g) })));
    const card = el("button", { class: "tg-card" + (r.l === TG.focus ? " focus" : ""), "data-s": c.isTruth ? "none" : r.s, title: "View full size" }, thumb, meta);
    card.addEventListener("click", () => openTubule(i));
    grid.append(card);
  });
  if (!TG.list.length) grid.append(el("p", { class: "meta", text: "No tubules in this group." }));
}
const closeGrid = () => { $("tg").hidden = true; };

// ---- single tubule
const TV = { i: 0, view: { s: 1, x: 0, y: 0 } };
function tvApply() { const v = TV.view; $("tv-stage").style.transform = `translate(${v.x}px, ${v.y}px) scale(${v.s})`; }
function tvReset() { TV.view = { s: 1, x: 0, y: 0 }; tvApply(); }
function openTubule(i) { TV.i = i; $("tv").hidden = false; renderTubule(); $("tv-close").focus(); }
function renderTubule() {
  const r = TG.list[TV.i], c = TG.ctx, t = r.t;
  tvReset();
  $("tv-img").src = "/ws/" + t.image;
  $("tv-outline").src = "/ws/" + t.outline;
  $("tv-eyebrow").textContent = c.section;
  $("tv-title").textContent = `Tubule #${r.l}`;
  const meta = $("tv-meta");
  meta.innerHTML = "";
  meta.append(el("div", {}, "Opened from ", el("b", { text: c.title })),
    el("div", { text: t.recommended ? "Accepted crop" : `Flagged crop${t.flags ? ": " + t.flags : ""}` }));
  const tb = $("tv-table");
  tb.innerHTML = "";
  tb.append(el("tr", {}, el("th", { text: "Source" }), el("th", { text: "Stage" })));
  tb.append(el("tr", { class: c.isTruth ? "current" : "" }, el("td", { text: c.truthName }), el("td", {}, el("b", { text: stageName(r.g) }))));
  const groups = new Set(c.peers.map((p) => p.group)).size;
  for (const p of c.peers) {
    const st = p.stages[r.l], sc = scoreOf(st, r.g);
    tb.append(el("tr", { class: !c.isTruth && p.title === c.title && p.group === c.group ? "current" : "", title: st ? SCORE_TEXT[sc] : "not staged" },
      el("td", {}, p.title, groups > 1 ? el("small", { text: p.group }) : ""), el("td", {}, el("i", { style: `background:${st ? SCORE_COLOUR[sc] : "transparent"}` }), stageName(st))));
  }
  $("tv-pos").textContent = `${TV.i + 1} / ${TG.list.length}`;
  $("tv-prev").disabled = TV.i === 0;
  $("tv-next").disabled = TV.i >= TG.list.length - 1;
}
function stepTubule(d) {
  const i = TV.i + d;
  if (i >= 0 && i < TG.list.length) { TV.i = i; renderTubule(); }
}
const closeTubule = () => { $("tv").hidden = true; const f = document.querySelectorAll("#tg-grid .tg-card")[TV.i]; if (f) f.focus(); };

function bindTubuleViews() {
  $("tg-close").addEventListener("click", closeGrid);
  for (const id of ["tg-stage", "tg-by", "tg-filter"]) $(id).addEventListener("change", () => { TG.focus = null; renderGrid(); });
  $("tg-outline").addEventListener("change", () => $("tg-grid").classList.toggle("no-outline", !$("tg-outline").checked));
  $("tg").addEventListener("click", (e) => { if (e.target === $("tg")) closeGrid(); });
  $("tv").addEventListener("click", (e) => { if (e.target === $("tv")) closeTubule(); });
  $("tv-close").addEventListener("click", closeTubule);
  $("tv-prev").addEventListener("click", () => stepTubule(-1));
  $("tv-next").addEventListener("click", () => stepTubule(1));
  const vw = $("tv-viewer");
  vw.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = vw.getBoundingClientRect(), v = TV.view;
    const s = Math.min(8, Math.max(1, v.s * Math.exp(-e.deltaY * 0.0015)));
    const px = e.clientX - r.left, py = e.clientY - r.top;
    v.x = px - ((px - v.x) * s) / v.s; v.y = py - ((py - v.y) * s) / v.s; v.s = s;
    if (s === 1) { v.x = 0; v.y = 0; }
    tvApply();
  }, { passive: false });
  let drag = null;
  vw.addEventListener("mousedown", (e) => { drag = { x: e.clientX - TV.view.x, y: e.clientY - TV.view.y }; vw.classList.add("dragging"); });
  window.addEventListener("mousemove", (e) => { if (drag) { TV.view.x = e.clientX - drag.x; TV.view.y = e.clientY - drag.y; tvApply(); } });
  window.addEventListener("mouseup", () => { drag = null; vw.classList.remove("dragging"); });
  vw.addEventListener("dblclick", tvReset);
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") { if (e.key === "Escape") e.target.blur(); else return; }
    if (!$("tv").hidden) {
      const k = e.key.toLowerCase();
      if (k === "escape") closeTubule();
      else if (k === "arrowleft") stepTubule(-1);
      else if (k === "arrowright") stepTubule(1);
      else if (k === "o") vw.classList.toggle("no-outline");
      else return;
      e.preventDefault();
    } else if (!$("tg").hidden && e.key === "Escape") { e.preventDefault(); closeGrid(); }
  });
}

function renderHelp() {
  const rows = review()
    ? [["1", "Stage I"], ["2 or 3", "Stage II–III"], ["4 … 9", "Stages IV … IX"], ["0", "Stage X"], ["-", "Stage XI"], ["=", "Stage XII"],
       ["B", "Can't stage / bad crop"]]
    : staging()
    ? [["1", "Stage I"], ["2 or 3", "Stage II–III"], ["4 … 9", "Stages IV … IX"], ["0", "Stage X"], ["-", "Stage XI"], ["=", "Stage XII"],
       ["B", "Can't stage / bad crop"], ["T", "Paper reading matches the handwriting"], ["F, then a stage key", "Correct the paper reading"]]
    : S.cfg.verdicts.map((v) => [v[2], v[1]]);
  rows.push(["→ or Space", "Skip to the next tubule"], ["←", "Go back one"], ["Delete", "Clear your label"], ["O", "Outline on/off"],
    ["Mouse wheel / drag", "Zoom / pan; double-click resets"]);
  const t = el("table");
  for (const [k, d] of rows) t.append(el("tr", {}, el("td", {}, el("kbd", { text: k })), el("td", { text: d })));
  $("help-keys").innerHTML = "";
  $("help-keys").append(t);
}

// ---------- boot ----------
async function boot() {
  S.cfg = await api("/api/config");
  $("version").textContent = "v" + S.cfg.version;
  $("backup-state").textContent = `Backups every ${S.cfg.backup_minutes} min`;
  S.tubules = await api("/api/tubules");
  S.byId = new Map(S.tubules.map((t) => [t.tubule_id, t]));
  $("start").addEventListener("click", login);
  $("name").addEventListener("keydown", (e) => { if (e.key === "Enter") login(); });
  $("logout").addEventListener("click", logout);
  $("prev").addEventListener("click", () => advance(-1));
  $("next").addEventListener("click", () => advance(1));
  $("clear").addEventListener("click", clearMine);
  $("paper-ok").addEventListener("click", confirmTranscription);
  $("paper-fix").addEventListener("click", () => { S.fixing = true; $("paper-fixing").hidden = false; });
  $("f-source").addEventListener("change", () => buildQueue());
  $("f-queue").addEventListener("change", () => buildQueue());
  $("ov-section").addEventListener("change", () => renderOverview());
  $("ov-show").addEventListener("change", () => renderOverview());
  for (const b of document.querySelectorAll(".ov-tab")) b.addEventListener("click", () => renderOverviewTab(b.dataset.tab));
  $("es-show").addEventListener("change", () => renderEyeSpy());
  $("es-comparison").addEventListener("change", () => renderEyeSpy());
  $("es-experiments").addEventListener("change", () => renderEyeSpy());
  bindSegGrid();
  bindSectionMap();
  bindTubuleViews();
  $("rv-blind").addEventListener("change", (e) => { S.blind = e.target.checked; store.set("blind", S.blind); renderSide(); });
  $("export").addEventListener("click", () => {
    window.location = "/api/export.csv?who=" + encodeURIComponent(S.session ? S.session.name : "manual");
    toast("CSV downloaded. A copy is also saved in the workspace exports folder.");
  });
  for (const b of document.querySelectorAll(".nav-item[data-view]")) b.addEventListener("click", () => setView(b.dataset.view));
  // Resume an open session after a reload (sessionStorage) or from #session=ID.
  const hashId = (location.hash.match(/session=(\d+)/) || [])[1];
  const savedId = hashId || (store.get("session") || {}).id;
  if (savedId) {
    try {
      await startSession(await api("/api/session?id=" + savedId));
      const v = (location.hash.match(/view=(\w+)/) || [])[1];
      const tab = (location.hash.match(/tab=(\w+)/) || [])[1]; // e.g. #view=overview&tab=eyespy
      if (tab) store.set("ov-tab", tab);
      if (v && $("view-" + v) && !(v === "overview" && !review())) setView(v);
      return;
    } catch { /* ended */ }
  }
  showLogin();
}
boot().catch((e) => { document.body.prepend(el("div", { class: "inline-error", text: "Could not start: " + e.message })); });
