// Greenroom UI: a small hash router and one render function per screen.
//   #/welcome        onboarding (first run)
//   #/home           greeting, start practice, due reviews, last session
//   #/characters     choose who to talk to
//   #/talk/<id>      the conversation (js/talk.js)
//   #/report/<id>    feedback for one session
//   #/progress       trends over sessions
//   #/review         flashcards
//   #/settings       profile, accent, model status, hands-free threshold
// Plain DOM + template strings, no framework: nothing to build, easy to read.

import { getJSON, sendJSON, api } from "./api.js";
import { lineChart } from "./charts.js";
import { startTalk } from "./talk.js";

const view = document.getElementById("view");
const topbar = document.getElementById("topbar");
export const state = { profile: null, status: null, reportCache: new Map(), leave: null };

// ---------- small helpers ----------
export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

export function toast(msg, ms = 3500) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { t.hidden = true; }, ms);
}

const GOAL_INFO = {
  placements: ["Campus placements", "HR and job interviews at college"],
  job: ["Work English", "Interviews, customers, meetings"],
  study_abroad: ["Study abroad", "Visa and admissions interviews"],
  general: ["Everyday English", "Relaxed conversation practice"],
};
const LEVEL_INFO = {
  A2: "I can say simple things slowly",
  B1: "I manage everyday topics",
  B2: "I speak fairly easily",
  C1: "I want to sound polished",
};
const ACCENT_INFO = {
  auto: ["Automatic", "Each character's own accent (or your destination's)"],
  american: ["American", "All characters speak American English"],
  british: ["British", "All characters speak British English"],
};
const DESTINATIONS = [["uk", "United Kingdom"], ["ireland", "Ireland"], ["australia", "Australia"],
  ["new zealand", "New Zealand"], ["usa", "United States"], ["canada", "Canada"], ["other", "Somewhere else"]];
export const ANALYZER_LABEL = { vocabulary: "Word choice", fluency: "Fluency", grammar: "Grammar" };

function fmtDate(ts) {
  return new Date(ts * 1000).toLocaleDateString(undefined, { day: "numeric", month: "short" });
}
const round = (v, d = 2) => (v === null || v === undefined ? "–" : Number(v).toFixed(d));

// ---------- router ----------
const routes = {
  welcome: renderOnboarding,
  home: renderHome,
  characters: renderCharacters,
  talk: (id) => startTalk(view, id),
  report: renderReport,
  progress: renderProgress,
  review: renderReview,
  settings: renderSettings,
};

async function route() {
  // Leaving a screen may need cleanup (the conversation stops the mic and saves).
  if (state.leave) { const leave = state.leave; state.leave = null; await leave(); }
  const [name, arg] = location.hash.replace(/^#\/?/, "").split("/");
  if (!state.profile && name !== "welcome") { location.hash = "#/welcome"; return; }
  const screen = routes[name] ? name : "home";
  topbar.hidden = screen === "welcome";
  document.querySelectorAll("[data-nav]").forEach((a) =>
    a.classList.toggle("active", a.dataset.nav === screen || (screen === "talk" && a.dataset.nav === "characters")));
  view.innerHTML = `<div class="empty"><span class="spinner"></span></div>`;
  try {
    await routes[screen](arg);
  } catch (err) {
    view.innerHTML = `<div class="banner error">${esc(err.message)}</div>`;
  }
  view.focus({ preventScroll: true });
}

// ---------- model status pills ----------
async function refreshStatus() {
  try {
    state.status = await getJSON("/api/status");
  } catch {
    state.status = null;
  }
  const s = state.status;
  const pill = (key, label) => {
    const st = s?.[key] ?? { state: "off", message: "Server not reachable" };
    return `<span class="pill" title="${esc(st.message || label + " ready")}"><i class="dot ${st.state}"></i><span class="label">${label}</span></span>`;
  };
  document.getElementById("status").innerHTML = pill("llm", "Coach") + pill("stt", "Mic") + pill("tts", "Voice");
  const loading = s && Object.values(s).some((v) => v.state === "loading");
  clearTimeout(refreshStatus._t);
  refreshStatus._t = setTimeout(refreshStatus, loading ? 2000 : 15000);
  return s;
}
export const getStatus = () => state.status;

// Banner text for features that are off, e.g. "Voice off: run setup".
function statusBanner(s) {
  if (!s) return "";
  const off = ["llm", "stt", "tts"].filter((k) => s[k].state === "off").map((k) => s[k].message);
  return off.length ? `<div class="banner">${off.map(esc).join("<br>")}</div>` : "";
}

// ---------- 1. onboarding + profile form (also used by Settings) ----------
function profileForm(p = {}) {
  const radio = (name, value, title, sub, checked) => `
    <label class="choice"><input type="radio" name="${name}" value="${value}" ${checked ? "checked" : ""} required>
      <span>${esc(title)}${sub ? `<small>${esc(sub)}</small>` : ""}</span></label>`;
  return `
    <div class="field"><label for="name">What should we call you?</label>
      <input type="text" id="name" name="name" maxlength="40" autocomplete="given-name" value="${esc(p.name || "")}" required></div>
    <div class="field"><label>What are you practising for?</label>
      <div class="choices">${Object.entries(GOAL_INFO).map(([v, [t, s]]) => radio("goal", v, t, s, p.goal === v)).join("")}</div></div>
    <div class="field" id="dest-field" ${p.goal === "study_abroad" ? "" : "hidden"}>
      <label for="destination">Where do you want to study?</label>
      <select id="destination" name="destination">
        ${DESTINATIONS.map(([v, t]) => `<option value="${v}" ${p.destination === v ? "selected" : ""}>${t}</option>`).join("")}
      </select>
      <div class="hint">With the automatic accent, characters speak like people there.</div></div>
    <div class="field"><label>Your English level</label>
      <div class="choices">${Object.entries(LEVEL_INFO).map(([v, s]) => radio("level", v, v, s, (p.level || "B1") === v)).join("")}</div>
      <div class="hint">The characters adjust how simply they speak. You can change this later.</div></div>
    <div class="field"><label>Accent you want to hear</label>
      <div class="choices">${Object.entries(ACCENT_INFO).map(([v, [t, s]]) => radio("accent", v, t, s, (p.accent || "auto") === v)).join("")}</div></div>`;
}

function readProfile(form, extra = {}) {
  const f = new FormData(form);
  const goal = f.get("goal");
  return {
    name: (f.get("name") || "").trim(), goal, level: f.get("level"), accent: f.get("accent"),
    destination: goal === "study_abroad" ? f.get("destination") : null,
    silence_ms: Number(f.get("silence_ms") || state.profile?.silence_ms || 1200), ...extra,
  };
}

function wireGoal(form) {
  form.addEventListener("change", (e) => {
    if (e.target.name === "goal") form.querySelector("#dest-field").hidden = e.target.value !== "study_abroad";
  });
}

async function renderOnboarding() {
  view.innerHTML = `
    <div class="stack" style="max-width:640px;margin:0 auto">
      <div class="hero">
        <img src="icon.svg" alt="" width="64" height="64">
        <h1>Welcome to Greenroom</h1>
        <p class="muted">A quiet place to rehearse your English before the real conversation.
        Everything stays on this computer.</p>
      </div>
      <form class="card" id="onb">${profileForm(state.profile || {})}
        <div class="row" style="margin-top:1.5rem;justify-content:flex-end">
          <button class="primary big" type="submit">Let's start</button></div>
      </form>
    </div>`;
  const form = document.getElementById("onb");
  wireGoal(form);
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      state.profile = (await sendJSON("/api/profile", readProfile(form), "PUT")).profile;
      location.hash = "#/home";
    } catch (err) { toast(err.message); }
  });
}

// ---------- 2. home ----------
function greeting() {
  const h = new Date().getHours();
  return h < 12 ? "Good morning" : h < 17 ? "Good afternoon" : "Good evening";
}

async function renderHome() {
  const [home, status] = await Promise.all([getJSON("/api/home"), refreshStatus()]);
  const last = home.last_session;
  const avg = last?.summary?.averages || {};
  view.innerHTML = `
    <div class="stack">
      ${statusBanner(status)}
      <section class="card hero">
        <h1>${greeting()}, ${esc(state.profile.name)}</h1>
        <p class="muted">${home.session_count ? "Ready for another round?" : "Your first conversation is one click away. Nobody is judging. It's a rehearsal."}</p>
        <a class="btn primary big" href="#/characters">Start practice</a>
      </section>
      <div class="grid">
        <a class="card" href="#/review" style="text-decoration:none;color:inherit">
          <h2>Review</h2>
          <p><b style="font-size:1.6rem">${home.due_cards}</b> <span class="muted">word${home.due_cards === 1 ? "" : "s"} due today</span></p>
          <span class="small muted">${home.due_cards ? "A few minutes of flashcards keeps new words in memory." : "Words from your sessions appear here."}</span>
        </a>
        <div class="card">
          <h2>Last session</h2>
          ${last ? `
            <p class="muted small">${fmtDate(last.started)} · ${esc(last.persona_name)} · ${last.summary.turns} turns</p>
            <div class="stats">
              <div class="stat"><b>${round(avg.vocabulary?.basic_word_count, 1)}</b><span>basic words / turn</span></div>
              <div class="stat"><b>${round(avg.fluency?.fillers_per_100_words, 1)}</b><span>fillers / 100 words</span></div>
            </div>
            <p style="margin-top:.75rem"><a href="#/report/${last.id}">See the full report</a></p>`
          : `<p class="muted">No sessions yet.</p>`}
        </div>
      </div>
    </div>`;
}

// ---------- 3. choose character ----------
async function renderCharacters() {
  const [personas, status] = await Promise.all([getJSON("/api/personas"), refreshStatus()]);
  const llmOff = status?.llm?.state === "off";
  view.innerHTML = `
    <div class="stack">
      <div><h1>Who do you want to talk to?</h1>
        <p class="muted">Characters for <b>${esc(GOAL_INFO[state.profile.goal]?.[0] || "you")}</b>, plus everyday practice.
        They stay in character and never correct you mid-conversation. Feedback comes at the end.</p></div>
      ${llmOff ? `<div class="banner error">${esc(status.llm.message)}</div>` : statusBanner(status)}
      <div class="grid">
        ${personas.map((p) => `
          <button class="card persona" data-id="${esc(p.id)}" ${llmOff ? "disabled" : ""}>
            <div class="avatar" aria-hidden="true">${esc(p.name.replace(/^(Mr|Ms|Mrs|Dr|Officer)\.?\s+/, "")[0])}</div>
            <h2>${esc(p.name)}</h2>
            <span class="muted">${esc(p.role)}</span>
            <span><span class="badge">${p.accent === "british" ? "British" : "American"} accent</span></span>
          </button>`).join("")}
      </div>
    </div>`;
  view.querySelectorAll(".persona").forEach((b) =>
    b.addEventListener("click", () => { location.hash = `#/talk/${b.dataset.id}`; }));
}

// ---------- 5. report ----------
// Split a sentence at every finding boundary so overlapping findings (e.g. a
// vocabulary span inside a grammar span) still each get highlighted.
function highlight(text, findings) {
  const cuts = new Set([0, text.length]);
  findings.forEach((f) => { cuts.add(f.start); cuts.add(f.end); });
  const pts = [...cuts].sort((a, b) => a - b);
  let html = "";
  for (let i = 0; i < pts.length - 1; i++) {
    const [a, b] = [pts[i], pts[i + 1]];
    const idx = findings.map((f, j) => (f.start <= a && f.end >= b ? j : -1)).filter((j) => j >= 0);
    const piece = esc(text.slice(a, b));
    html += idx.length
      ? `<mark class="${findings[idx[0]].analyzer}" tabindex="0" data-f="${idx.join(",")}">${piece}</mark>`
      : piece;
  }
  return html;
}

function popoverHTML(findings) {
  return findings.map((f) => `
    <div><b>${esc(ANALYZER_LABEL[f.analyzer] || f.analyzer)}:</b> ${esc(f.message)}
      ${f.suggestions?.length ? `<div class="try">Try: ${f.suggestions.map((s) => `<b>${esc(s)}</b>`).join(", ")}</div>` : ""}
    </div>`).join("<hr style='border:none;border-top:1px solid var(--line)'>");
}

// One popover at a time, shared by all reports; any click elsewhere closes it.
let pop = null;
function closePopover() {
  pop?.remove();
  pop = null;
  document.querySelectorAll("mark.open").forEach((m) => m.classList.remove("open"));
}
document.addEventListener("click", closePopover);
window.addEventListener("hashchange", closePopover);

function openPopover(mark, turns) {
  closePopover();
  const turn = turns[Number(mark.closest("[data-turn]").dataset.turn)];
  const fs = mark.dataset.f.split(",").map((j) => turn.findings[Number(j)]);
  pop = document.createElement("div");
  pop.className = "popover";
  pop.innerHTML = popoverHTML(fs);
  document.body.appendChild(pop);
  const r = mark.getBoundingClientRect();
  const maxLeft = document.documentElement.clientWidth - pop.offsetWidth - 8;
  pop.style.left = `${window.scrollX + Math.max(8, Math.min(r.left, maxLeft))}px`;
  pop.style.top = `${window.scrollY + r.bottom + 6}px`;
  mark.classList.add("open");
}

// Hover (mouse), tap (touch) and keyboard focus all open the explanation.
function wirePopovers(root, turns) {
  root.addEventListener("mouseover", (e) => {
    const m = e.target.closest("mark");
    if (m && !m.classList.contains("open")) openPopover(m, turns);
  });
  root.addEventListener("click", (e) => {
    const m = e.target.closest("mark");
    if (!m) return;
    e.stopPropagation();   // keep the document handler from closing it right away
    if (m.classList.contains("open") && e.pointerType !== "mouse") closePopover();
    else openPopover(m, turns);
  });
  root.addEventListener("focusin", (e) => {
    const m = e.target.closest("mark");
    if (m) openPopover(m, turns);
  });
}

async function renderReport(id) {
  let r = state.reportCache.get(String(id));
  if (!r) r = await getJSON(`/api/reports/${id}`);
  const avg = r.summary.averages || {};
  const total = r.turns.reduce((n, t) => n + t.findings.length, 0);
  const lat = r.latency || {};
  view.innerHTML = `
    <div class="stack">
      <div class="row spread">
        <div><h1>Your feedback</h1>
          <p class="muted">${fmtDate(r.started)} · ${esc(r.persona_name)} · ${r.summary.turns} turns · ${Math.round((r.duration_s || 0) / 60)} min</p></div>
        <div class="row"><a class="btn" href="#/progress">Progress</a><a class="btn primary" href="#/characters">Practise again</a></div>
      </div>
      <section class="card">
        <div class="row spread"><h2>What you said</h2>
          <div class="legend"><span><i style="background:var(--vocab)"></i>Word choice</span>
            <span><i style="background:var(--fluency)"></i>Fluency</span><span><i style="background:var(--grammar)"></i>Grammar</span></div></div>
        <p class="muted small">${total ? "Tap or hover a highlighted part to see why and what to try instead." : "Nothing flagged. Clear, varied English. Well done!"}</p>
        <ol class="turns">${r.turns.map((t, i) => `<li class="sentence" data-turn="${i}">${highlight(t.text, t.findings)}</li>`).join("")}</ol>
      </section>
      ${r.exercises.length ? `
      <section class="stack"><h2>Practise next</h2>
        ${r.exercises.map((x) => `<div class="card exercise"><h3>${esc(x.title)}</h3><p>${esc(x.prompt)}</p>
          ${x.tip ? `<p class="muted small">${esc(x.tip)}</p>` : ""}</div>`).join("")}
      </section>` : ""}
      <section class="card"><h2>Numbers</h2>
        <div class="stats">
          <div class="stat"><b>${round(avg.vocabulary?.basic_word_count, 1)}</b><span>basic words per turn</span></div>
          <div class="stat"><b>${round(avg.vocabulary?.type_token_ratio, 2)}</b><span>word variety (type-token ratio)</span></div>
          <div class="stat"><b>${round(avg.fluency?.fillers_per_100_words, 1)}</b><span>fillers per 100 words</span></div>
          <div class="stat"><b>${round(avg.fluency?.avg_sentence_length, 1)}</b><span>words per sentence</span></div>
          ${avg.grammar ? `<div class="stat"><b>${round(avg.grammar.error_count, 1)}</b><span>grammar fixes per turn</span></div>` : ""}
          ${lat.total_s !== undefined ? `<div class="stat"><b>${round(lat.total_s, 2)} s</b><span>coach response time</span></div>` : ""}
        </div>
      </section>
    </div>`;
  wirePopovers(view.querySelector("ol.turns"), r.turns);   // fresh element each render, so listeners never pile up
}

// ---------- 6. progress ----------
async function renderProgress() {
  const p = await getJSON("/api/progress");
  if (!p.sessions.length) {
    view.innerHTML = `<div class="card empty"><h2>No sessions yet</h2><p>Your trends appear after your first conversation.</p>
      <a class="btn primary" href="#/characters">Start practice</a></div>`;
    return;
  }
  const series = (key) => p.sessions.map((s) => ({ label: fmtDate(s.started), value: s[key] }));
  const chart = (title, key, hint, opts) => `<div class="card chart"><h3>${title}</h3><p class="muted small">${hint}</p>${lineChart(series(key), opts)}</div>`;
  const maxCount = Math.max(1, ...p.most_flagged.map((w) => w.count));
  view.innerHTML = `
    <div class="stack">
      <h1>Your progress</h1>
      <p class="muted">${p.sessions.length} session${p.sessions.length === 1 ? "" : "s"} so far.</p>
      <div class="grid">
        ${chart("Basic words per turn", "basic_words_per_turn", "Lower is better: fewer “good”, “nice”, “thing”.", { decimals: 1 })}
        ${chart("Word variety", "type_token_ratio", "Higher is better: share of different content words.", { decimals: 2 })}
        ${chart("Fillers per 100 words", "fillers_per_100_words", "Lower is better: fewer “um”, “you know”.", { decimals: 1 })}
        ${p.sessions.some((s) => s.grammar_errors_per_turn != null) ? chart("Grammar fixes per turn", "grammar_errors_per_turn", "Lower is better.", { decimals: 1 }) : ""}
        ${chart("Coach response time", "latency_total_s", "From the end of your speech to the coach's first word.", { decimals: 2, unit: "s" })}
      </div>
      <section class="card"><h2>Most-flagged words</h2>
        ${p.most_flagged.length ? `<div class="bars">${p.most_flagged.map((w) => `
          <div class="bar"><span>${esc(w.text)}</span><span class="track"><span class="fill" style="width:${(100 * w.count) / maxCount}%;display:block"></span></span><b>${w.count}</b></div>`).join("")}</div>`
          : `<p class="muted">Nothing flagged yet.</p>`}
      </section>
      <section class="card"><h2>All sessions</h2>
        <ol class="turns" reversed>${[...p.sessions].reverse().map((s) => `<li><a href="#/report/${s.id}">${fmtDate(s.started)}: ${esc(s.persona_name)}</a> <span class="muted small">(${s.turns} turns)</span></li>`).join("")}</ol>
      </section>
    </div>`;
}

// ---------- 7. review ----------
async function renderReview() {
  const { cards } = await getJSON("/api/review/due");
  let i = 0;
  const show = () => {
    if (i >= cards.length) {
      view.innerHTML = `<div class="card empty"><h2>${cards.length ? "All done for today" : "Nothing to review"}</h2>
        <p>${cards.length ? `You reviewed ${cards.length} word${cards.length === 1 ? "" : "s"}. They'll come back when it's time.` : "Words flagged in your sessions become flashcards here."}</p>
        <a class="btn primary" href="#/home">Back home</a></div>`;
      return;
    }
    const c = cards[i];
    view.innerHTML = `
      <div class="stack" style="max-width:560px;margin:0 auto">
        <p class="muted">Card ${i + 1} of ${cards.length}</p>
        <div class="card flash">
          <p class="muted small">You said</p>
          <div class="front">“${esc(c.front)}”</div>
          ${c.example ? `<p class="example">${esc(c.example)}</p>` : ""}
          <p class="muted">Which stronger word could you use? Say it out loud, then reveal.</p>
          <div id="back" hidden>
            <p style="font-size:1.3rem"><b>${(c.suggestions || []).map(esc).join(", ")}</b></p>
            <p class="muted small">${esc(c.message || "")}</p>
          </div>
          <button class="primary" id="reveal">Show answer</button>
        </div>
        <div class="grades" id="grades" hidden>
          <button data-g="0">Again<small>forgot</small></button>
          <button data-g="3">Hard<small>remembered slowly</small></button>
          <button data-g="4">Good<small>remembered</small></button>
          <button data-g="5">Easy<small>instantly</small></button>
        </div>
      </div>`;
    view.querySelector("#reveal").onclick = (e) => {
      e.target.hidden = true;
      view.querySelector("#back").hidden = false;
      view.querySelector("#grades").hidden = false;
    };
    view.querySelectorAll("[data-g]").forEach((b) => (b.onclick = async () => {
      try {
        await sendJSON(`/api/review/${c.id}`, { grade: Number(b.dataset.g) });
        i += 1;
        show();
      } catch (err) { toast(err.message); }
    }));
  };
  show();
}

// ---------- 8. settings ----------
async function renderSettings() {
  const status = await refreshStatus();
  const p = state.profile;
  const row = (key, label) => {
    const s = status?.[key] || { state: "off", message: "Server not reachable" };
    return `<li class="row"><i class="dot ${s.state}"></i><b>${label}</b><span class="muted small">${s.state === "ready" ? "Ready" + (s.model ? ` (${esc(s.model)})` : "") : esc(s.message)}</span></li>`;
  };
  view.innerHTML = `
    <div class="stack" style="max-width:680px">
      <h1>Settings</h1>
      <form class="card" id="settings">
        <h2>Profile</h2>
        ${profileForm(p)}
        <div class="field"><label for="silence">Hands-free: pause before your turn ends</label>
          <input type="range" id="silence" name="silence_ms" min="600" max="3000" step="100" value="${p.silence_ms || 1200}" style="width:100%">
          <div class="hint"><span id="silence-val">${((p.silence_ms || 1200) / 1000).toFixed(1)}</span> seconds of silence.
          Longer if you often pause mid-sentence; shorter for faster replies.</div></div>
        <div class="row" style="margin-top:1.25rem;justify-content:flex-end"><button class="primary" type="submit">Save</button></div>
      </form>
      <section class="card"><h2>Models</h2>
        <ul style="list-style:none;padding:0;margin:0;display:grid;gap:.5rem">
          ${row("llm", "Coach (LLM)")}${row("stt", "Speech recognition")}${row("tts", "Coach voice")}
          ${row("bert", "Smart word suggestions (BERT)")}${row("grammar", "Grammar feedback (T5)")}
        </ul>
        <p class="muted small" style="margin-top:.75rem">Everything runs on this computer. Missing models? Run <code>python setup_models.py</code> once.</p>
      </section>
    </div>`;
  const form = document.getElementById("settings");
  wireGoal(form);
  form.silence_ms.addEventListener("input", (e) => {
    document.getElementById("silence-val").textContent = (e.target.value / 1000).toFixed(1);
  });
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      state.profile = (await sendJSON("/api/profile", readProfile(form), "PUT")).profile;
      toast("Saved.");
    } catch (err) { toast(err.message); }
  });
}

// ---------- boot ----------
async function boot() {
  try {
    state.profile = (await getJSON("/api/profile")).profile;
  } catch (err) {
    view.innerHTML = `<div class="banner error">Can't reach the Greenroom server: ${esc(err.message)}</div>`;
    return;
  }
  refreshStatus();
  window.addEventListener("hashchange", route);
  if (!location.hash) location.hash = state.profile ? "#/home" : "#/welcome";
  else route();
}

export { api, refreshStatus };
boot();
