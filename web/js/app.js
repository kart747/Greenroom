// Greenroom UI: a small hash router and one render function per screen,
// inside a chat-app layout (sidebar with recent conversations + main area).
//   #/welcome        onboarding (first run, no sidebar)
//   #/home           start page: ask Sam anything, suggestions, roleplays
//   #/tutor/new      a new chat with Sam, the tutor (js/talk.js)
//   #/tutor/<id>     continue a saved tutor chat
//   #/characters     choose a roleplay character
//   #/talk/<id>      a roleplay conversation (js/talk.js)
//   #/report/<id>    feedback for one session
//   #/progress       trends over sessions
//   #/review         flashcards
//   #/settings       profile, accent, model status, hands-free threshold
// Plain DOM + template strings, no framework: nothing to build, easy to read.

import { getJSON, sendJSON, api } from "./api.js";
import { lineChart } from "./charts.js";
import { showFindings } from "./feedback.js";
import { startTalk } from "./talk.js";

const view = document.getElementById("view");
const sidebar = document.getElementById("sidebar");
// pendingMessage: typed on the start page; the new tutor chat sends it first.
export const state = { profile: null, status: null, reportCache: new Map(), leave: null, pendingMessage: null };

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
// In-chat feedback, chosen by the learner (server: FEEDBACK_POLICIES).
export const FEEDBACK_INFO = {
  after: ["After the session", "Focus on talking; see everything in the report"],
  tutor_only: ["Only with Sam", "Markers in tutor chats; roleplays stay clean"],
  gentle: ["Gently, in every chat", "Your most important mistake per message"],
  all: ["Every mistake", "All markers, in every chat"],
};
const CORRECTION_INFO = {
  gentle: ["Gentle", "Only the most important mistake in each message"],
  thorough: ["Thorough", "Every clear mistake (up to three per message)"],
  off: ["Off", "Only when I ask"],
};
const DESTINATIONS = [["uk", "United Kingdom"], ["ireland", "Ireland"], ["australia", "Australia"],
  ["new zealand", "New Zealand"], ["usa", "United States"], ["canada", "Canada"], ["other", "Somewhere else"]];

function fmtDate(ts) {
  return new Date(ts * 1000).toLocaleDateString(undefined, { day: "numeric", month: "short" });
}
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
const round = (v, d = 2) => (v === null || v === undefined ? "–" : Number(v).toFixed(d));

// ---------- router ----------
const routes = {
  welcome: renderOnboarding,
  home: renderHome,
  characters: renderCharacters,
  tutor: (id) => {
    const firstMessage = state.pendingMessage;
    state.pendingMessage = null;
    return startTalk(view, id === "new" ? { mode: "tutor", firstMessage } : { mode: "tutor", resumeId: Number(id) });
  },
  talk: (id) => startTalk(view, { mode: "roleplay", personaId: id }),
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
  document.body.classList.toggle("bare", screen === "welcome");
  document.body.dataset.screen = screen;      // lets the chat screen use the full height
  closeSidebar();
  document.querySelectorAll("[data-nav]").forEach((a) =>
    a.classList.toggle("active", a.dataset.nav === screen
      || (screen === "talk" && a.dataset.nav === "characters")
      || (screen === "tutor" && arg === "new" && a.dataset.nav === "tutor")));
  markRecent();
  view.innerHTML = `<div class="empty"><span class="spinner"></span></div>`;
  try {
    await routes[screen](arg);
  } catch (err) {
    view.innerHTML = `<div class="banner error">${esc(err.message)}</div>`;
  }
  view.focus({ preventScroll: true });
}

// ---------- sidebar: recent conversations (like ChatGPT's history) ----------
export async function refreshSidebar() {
  let convs = [];
  let due = 0;
  try {
    [convs, { due_cards: due }] = await Promise.all([getJSON("/api/conversations"), getJSON("/api/home")]);
  } catch { /* server busy; keep the old list */ return; }
  const recent = document.getElementById("recent");
  recent.innerHTML = convs.length ? convs.map((c) => {
    // Tutor chats reopen and continue; roleplays open their report.
    const href = c.mode === "tutor" ? `#/tutor/${c.id}` : `#/report/${c.id}`;
    const title = c.mode === "tutor" ? (c.title || "Chat with Sam") : c.persona_name;
    const sub = `${c.mode === "tutor" ? "Tutor" : "Roleplay"} · ${fmtDate(c.updated)}`;
    return `<a href="${href}" title="${esc(title)}">${esc(title)}<small>${sub}</small></a>`;
  }).join("") : `<div class="note">Your conversations will appear here.</div>`;
  const count = document.getElementById("due-count");
  count.hidden = !due;
  count.textContent = due;
  markRecent();
}

function markRecent() {
  document.querySelectorAll("#recent a").forEach((a) => a.classList.toggle("active", a.getAttribute("href") === location.hash));
}

function openSidebar() {
  sidebar.classList.add("open");
  document.getElementById("scrim").hidden = false;
}
function closeSidebar() {
  sidebar.classList.remove("open");
  document.getElementById("scrim").hidden = true;
}
document.getElementById("open-side").addEventListener("click", openSidebar);
document.getElementById("close-side").addEventListener("click", closeSidebar);
document.getElementById("scrim").addEventListener("click", closeSidebar);

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
    silence_ms: Number(f.get("silence_ms") || state.profile?.silence_ms || 1200),
    tutor_corrections: f.get("tutor_corrections") || state.profile?.tutor_corrections || "gentle",
    chat_feedback: f.get("chat_feedback") || state.profile?.chat_feedback || "after", ...extra,
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

// ---------- 2. home: the start page ----------
function greeting() {
  const h = new Date().getHours();
  return h < 12 ? "Good morning" : h < 17 ? "Good afternoon" : "Good evening";
}

// Conversation starters for Sam, by the learner's goal.
const STARTERS = {
  placements: ["Help me answer “Tell me about yourself”", "What's wrong with “I am having 2 years experience”?",
    "How do I describe my final-year project?", "Ask me common HR questions"],
  job: ["How do I politely say no to a customer?", "Useful phrases for team meetings",
    "How do I apologise professionally?", "Practise small talk with a new colleague"],
  study_abroad: ["How do I explain why I chose this university?", "What might a visa officer ask me?",
    "Help me talk about how I'll fund my studies", "Phrases for my first day on campus"],
  general: ["Explain “since” vs “for”", "Let's chat about my weekend",
    "How can I sound more fluent?", "Teach me 3 phrases for small talk"],
};

async function renderHome() {
  const [home, personas, status] = await Promise.all([getJSON("/api/home"), getJSON("/api/personas"), refreshStatus()]);
  const starters = STARTERS[state.profile.goal] || STARTERS.general;
  const llmOff = status?.llm?.state === "off";
  view.innerHTML = `
    <div class="landing stack">
      ${statusBanner(status)}
      <div class="landing-head">
        <img src="icon.svg" alt="" width="52" height="52">
        <h1>${greeting()}, ${esc(state.profile.name)}</h1>
        <p class="muted">Ask Sam, your English tutor, anything, or just start talking. Nobody is judging. It's a rehearsal.</p>
      </div>
      <form class="composer" id="start-chat">
        <textarea id="start-text" rows="1" placeholder="Ask Sam anything about English, or say hi…" aria-label="Message Sam" ${llmOff ? "disabled" : ""}></textarea>
        <button class="primary" type="submit" ${llmOff ? "disabled" : ""}>Send</button>
      </form>
      <div class="chips">${starters.map((t) => `<button class="chip" type="button">${esc(t)}</button>`).join("")}
        <a class="chip" href="#/tutor/new">Just start a voice chat</a></div>
      <section class="stack">
        <h2>Or practise a real situation</h2>
        <p class="muted small">Roleplay characters stay in character and never correct you; you get feedback at the end.</p>
        <div class="grid">${personas.map((p) => personaCard(p, llmOff)).join("")}
          <div class="card persona own" role="button" tabindex="0" data-id="new">
            <div class="avatar" aria-hidden="true">＋</div><h3>Your own situation</h3>
            <span class="muted small">Describe it, or paste a job description, and practise it</span></div></div>
      </section>
      <div class="grid">
        <a class="card link-card" href="#/review"><h3>Review</h3>
          <p><b class="big-num">${home.due_cards}</b> <span class="muted">word${home.due_cards === 1 ? "" : "s"} due</span></p></a>
        ${home.last_session ? `<a class="card link-card" href="#/report/${home.last_session.id}"><h3>Last report</h3>
          <p class="muted small">${fmtDate(home.last_session.started)} · ${esc(home.last_session.persona_name)} · ${plural(home.last_session.summary.turns, "turn")}</p></a>` : ""}
      </div>
    </div>`;
  const start = (text) => {
    if (!text.trim()) return;
    state.pendingMessage = text.trim();
    location.hash = "#/tutor/new";
  };
  const ta = view.querySelector("#start-text");
  autoGrow(ta);
  ta.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); start(ta.value); }
  });
  view.querySelector("#start-chat").addEventListener("submit", (e) => { e.preventDefault(); start(ta.value); });
  view.querySelectorAll("button.chip").forEach((c) => c.addEventListener("click", () => start(c.textContent)));
  wirePersonaCards();
  ta.focus();
}

// A textarea that grows with its text, up to a limit (used by every composer).
export function autoGrow(ta, max = 180) {
  const fit = () => { ta.style.height = "auto"; ta.style.height = `${Math.min(ta.scrollHeight, max)}px`; };
  ta.addEventListener("input", fit);
  fit();
}

function personaCard(p, disabled, deletable = false) {
  const badge = p.kind === "job" ? "Job interview" : p.kind ? "Your situation" : `${p.accent === "british" ? "British" : "American"} accent`;
  return `<div class="card persona" role="button" tabindex="0" data-id="${esc(p.id)}" ${disabled ? 'aria-disabled="true"' : ""}>
    ${deletable ? `<button class="icon-btn persona-del" data-delete="${esc(p.id.replace("custom_", ""))}" aria-label="Delete ${esc(p.name)}">✕</button>` : ""}
    <div class="avatar" aria-hidden="true">${esc(p.name.replace(/^(Mr|Ms|Mrs|Dr|Officer)\.?\s+/, "")[0])}</div>
    <h3>${esc(p.name)}</h3>
    <span class="muted small">${esc(p.role)}</span>
    <span><span class="badge">${esc(badge)}</span></span>
  </div>`;
}

function wirePersonaCards() {
  view.querySelectorAll(".persona[data-id]").forEach((card) => {
    if (card.getAttribute("aria-disabled") === "true") return;
    const go = () => { location.hash = card.dataset.id === "new" ? "#/characters" : `#/talk/${card.dataset.id}`; };
    card.addEventListener("click", go);
    card.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
  });
}

// ---------- 3. choose character ----------
async function renderCharacters() {
  const [personas, mine, status] = await Promise.all([getJSON("/api/personas"), getJSON("/api/scenarios"), refreshStatus()]);
  const llmOff = status?.llm?.state === "off";
  view.innerHTML = `
    <div class="stack">
      <div><h1>Roleplay a situation</h1>
        <p class="muted">Characters for <b>${esc(GOAL_INFO[state.profile.goal]?.[0] || "you")}</b>, plus everyday practice.
        They stay in character and never correct you mid-conversation. Feedback comes at the end.
        Want corrections as you go? <a href="#/tutor/new">Chat with Sam</a> instead.</p></div>
      ${llmOff ? `<div class="banner error">${esc(status.llm.message)}</div>` : statusBanner(status)}
      <section class="card create-scenario">
        <h2>Practise your own situation</h2>
        <div class="row" role="tablist">
          <button class="chip active" type="button" data-kind="situation" role="tab" aria-selected="true">Describe a situation</button>
          <button class="chip" type="button" data-kind="job" role="tab" aria-selected="false">Paste a job description</button>
        </div>
        <textarea id="sc-text" rows="3" maxlength="3000" placeholder="${esc(SCENARIO_HINT.situation)}"></textarea>
        <div class="row"><button class="primary" id="sc-create" type="button" ${llmOff ? "disabled" : ""}>Create the character</button>
          <span class="muted small" id="sc-status">Greenroom invents the other person; you start talking right away.</span></div>
      </section>
      ${mine.length ? `<section class="stack"><h2>Your scenarios</h2>
        <div class="grid">${mine.map((p) => personaCard(p, llmOff, true)).join("")}</div></section>` : ""}
      <h2>Ready-made characters</h2>
      <div class="grid">${personas.map((p) => personaCard(p, llmOff)).join("")}</div>
    </div>`;
  wirePersonaCards();
  let kind = "situation";
  const text = view.querySelector("#sc-text");
  view.querySelectorAll("[data-kind]").forEach((tab) => tab.addEventListener("click", () => {
    kind = tab.dataset.kind;
    view.querySelectorAll("[data-kind]").forEach((t) => {
      t.classList.toggle("active", t === tab);
      t.setAttribute("aria-selected", String(t === tab));
    });
    text.placeholder = SCENARIO_HINT[kind];
    text.rows = kind === "job" ? 8 : 3;
    text.focus();
  }));
  view.querySelector("#sc-create").addEventListener("click", async (e) => {
    const btn = e.target;
    btn.disabled = true;
    view.querySelector("#sc-status").innerHTML = `<span class="spinner"></span> Designing your character…`;
    try {
      const p = await sendJSON("/api/scenarios", { text: text.value, kind });
      location.hash = `#/talk/${p.id}`;          // straight into the conversation
    } catch (err) {
      view.querySelector("#sc-status").textContent = err.message;
      btn.disabled = false;
    }
  });
  view.querySelectorAll("[data-delete]").forEach((b) => b.addEventListener("click", async (e) => {
    e.stopPropagation();
    await api(`/api/scenarios/${b.dataset.delete}`, { method: "DELETE" });
    renderCharacters();
  }));
}

const SCENARIO_HINT = {
  situation: "e.g. I need to ask my manager for two days of leave for my sister's wedding.",
  job: "Paste the job description: title, responsibilities, skills…",
};

// ---------- 5. report ----------
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
          <p class="muted">${fmtDate(r.started)} · ${esc(r.persona_name)} · ${plural(r.summary.turns, "turn")} · ${Math.max(1, Math.round((r.duration_s || 0) / 60))} min</p></div>
        <div class="row"><a class="btn" href="#/progress">Progress</a>
          ${r.mode === "tutor" ? `<a class="btn primary" href="#/tutor/${id}">Continue this chat</a>`
                               : `<a class="btn primary" href="#/characters">Practise again</a>`}</div>
      </div>
      <section class="card">
        <div class="row spread"><h2>What you said</h2>
          <div class="legend"><span><i style="background:var(--vocab)"></i>Word choice</span>
            <span><i style="background:var(--fluency)"></i>Fluency &amp; pace</span><span><i style="background:var(--grammar)"></i>Grammar</span></div></div>
        <p class="muted small">${total ? "Tap or hover a highlighted part to see why and what to try instead." : "Nothing flagged. Clear, varied English. Well done!"}</p>
        <ol class="turns">${r.turns.map(() => `<li class="sentence"></li>`).join("")}</ol>
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
          ${avg.pace?.words_per_minute != null ? `<div class="stat"><b>${round(avg.pace.words_per_minute, 0)}</b><span>words per minute (conversation: about 120–160)</span></div>
            <div class="stat"><b>${round(avg.pace.pauses_per_minute, 1)}</b><span>pauses per minute (250 ms or longer)</span></div>` : ""}
          ${lat.total_s !== undefined ? `<div class="stat"><b>${round(lat.total_s, 2)} s</b><span>coach response time</span></div>` : ""}
        </div>
      </section>
    </div>`;
  view.querySelectorAll("ol.turns > li").forEach((li, i) => showFindings(li, r.turns[i].text, r.turns[i].findings));
}

// ---------- 6. progress ----------
async function renderProgress() {
  const p = await getJSON("/api/progress");
  if (!p.sessions.length) {
    view.innerHTML = `<div class="card empty"><h2>No sessions yet</h2><p>Your trends appear after your first conversation.</p>
      <a class="btn primary" href="#/home">Start practising</a></div>`;
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
        ${p.sessions.some((s) => s.words_per_minute != null) ? chart("Speaking speed", "words_per_minute", "Words per minute when you speak; about 120–160 is a comfortable conversation pace.", { decimals: 0 }) : ""}
        ${chart("Coach response time", "latency_total_s", "From the end of your speech to the coach's first word.", { decimals: 2, unit: "s" })}
      </div>
      <section class="card"><h2>Most-flagged words</h2>
        ${p.most_flagged.length ? `<div class="bars">${p.most_flagged.map((w) => `
          <div class="bar"><span>${esc(w.text)}</span><span class="track"><span class="fill" style="width:${(100 * w.count) / maxCount}%;display:block"></span></span><b>${w.count}</b></div>`).join("")}</div>`
          : `<p class="muted">Nothing flagged yet.</p>`}
      </section>
      <section class="card"><h2>All sessions</h2>
        <ol class="turns" reversed>${[...p.sessions].reverse().map((s) => `<li><a href="#/report/${s.id}">${fmtDate(s.started)}: ${esc(s.persona_name)}</a> <span class="muted small">(${plural(s.turns, "turn")})</span></li>`).join("")}</ol>
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
        <div class="field"><label>Feedback during chats</label>
          <div class="choices">${Object.entries(FEEDBACK_INFO).map(([v, [t, sub]]) => `
            <label class="choice"><input type="radio" name="chat_feedback" value="${v}" ${(p.chat_feedback || "after") === v ? "checked" : ""}>
              <span>${esc(t)}<small>${esc(sub)}</small></span></label>`).join("")}</div>
          <div class="hint">Feedback shows as a small marker under <b>your</b> message. Characters always stay in character; you can also switch this inside a chat.</div></div>
        <div class="field"><label>When you chat with Sam, Sam's own corrections should be…</label>
          <div class="choices">${Object.entries(CORRECTION_INFO).map(([v, [t, sub]]) => `
            <label class="choice"><input type="radio" name="tutor_corrections" value="${v}" ${(p.tutor_corrections || "gentle") === v ? "checked" : ""}>
              <span>${esc(t)}<small>${esc(sub)}</small></span></label>`).join("")}</div>
          <div class="hint">Roleplay characters never correct you; their feedback is in the report.</div></div>
        <div class="row" style="margin-top:1.25rem;justify-content:flex-end"><button class="primary" type="submit">Save</button></div>
      </form>
      <section class="card" id="memory"><h2>What Sam remembers</h2>
        <p class="muted small">After each chat, Sam notes a few facts you mentioned, so the next chat can pick up from there. They stay on this computer.</p>
        <ul class="memory-list" id="memory-list"></ul>
        <button class="ghost danger small" id="forget-all" type="button">Forget everything</button>
      </section>
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
  const showMemory = (facts) => {
    const list = document.getElementById("memory-list");
    list.innerHTML = facts.length
      ? facts.map((f) => `<li><span>${esc(f.text)}</span><button class="icon-btn" data-id="${f.id}" aria-label="Forget this">✕</button></li>`).join("")
      : `<li class="muted small">Nothing yet. Chat with Sam and tell it about yourself.</li>`;
    document.getElementById("forget-all").hidden = !facts.length;
  };
  showMemory((await getJSON("/api/memory")).facts);
  document.getElementById("memory-list").addEventListener("click", async (e) => {
    const b = e.target.closest("button[data-id]");
    if (b) showMemory((await (await api(`/api/memory/${b.dataset.id}`, { method: "DELETE" })).json()).facts);
  });
  document.getElementById("forget-all").addEventListener("click", async () => {
    showMemory((await (await api("/api/memory", { method: "DELETE" })).json()).facts);
    toast("Sam has forgotten everything.");
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
  document.getElementById("sidebar").hidden = false;
  refreshSidebar();
  window.addEventListener("hashchange", route);
  if (!location.hash) location.hash = state.profile ? "#/home" : "#/welcome";
  else route();
}

export { api, refreshStatus };
boot();
