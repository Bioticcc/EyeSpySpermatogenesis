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
  try { await loadLabels(); if (!$("view-annotate").hidden) renderSide(); } catch { /* retry later */ }
}, 20000);

const labelsOf = (tid) => S.labels.get(tid) || [];
const mineOf = (tid) => labelsOf(tid).find((l) => l.origin === "web" && l.annotator === S.session.name);
const paperOf = (tid) => labelsOf(tid).find((l) => l.origin === "paper");
const othersOf = (tid) => labelsOf(tid).filter((l) => !(l.origin === "web" && l.annotator === S.session.name));
const staging = () => S.session && S.session.mode === "staging";
const review = () => S.session && S.session.mode === "model_review";
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
  if (name === "overview") renderOverview().catch((e) => toast("Overview: " + e.message, true));
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
  for (const r of bySrc.values()) {
    const cells = st ? [r.name, r.n, r.any, r.two, r.paper, r.check, r.mine] : [r.name, r.n, r.any, r.two, r.mine];
    tb.append(el("tr", {}, ...cells.map((c) => el("td", { text: String(c) }))));
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
  const card = (title, src, stat) => {
    const img = el("img", { src, alt: title, loading: "lazy", onclick: () => window.open(src, "_blank") });
    return el("div", { class: "ov-card" }, el("h3", { text: title }), img, el("div", { class: "stat", text: stat }));
  };
  for (const r of runs) {
    const s = r.sections.find((x) => x.section === sec);
    if (!s) continue;
    const cards = el("div", { class: "ov-cards", style: `--cols:${ncols}` });
    cards.append(card("Paper stage (Yeongseok)", `/ws/review/${r.run}/${s.paper}`, "Reference: unchecked transcription of the hand-staged sheet"));
    for (const [name, m] of Object.entries(s.models)) {
      cards.append(card(name, `/ws/review/${r.run}/${show === "agree" ? m.agree : m.stage}`,
        `This section: exact ${pct(m.exact)}, within one ${pct(m.within1)} (n=${m.n})`));
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
      if (v && $("view-" + v) && !(v === "overview" && !review())) setView(v);
      return;
    } catch { /* ended */ }
  }
  showLogin();
}
boot().catch((e) => { document.body.prepend(el("div", { class: "inline-error", text: "Could not start: " + e.message })); });
