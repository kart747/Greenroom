// The conversation screen: Tutor chats with Sam and roleplays with characters.
//
// Two ways to talk:
//   push-to-talk  hold the mic button (or Space); release to send
//   hands-free    toggle on; the voice-activity detector in vad.js starts and
//                 stops recording by itself, and speaking while the coach
//                 talks interrupts it (barge-in)
// ...or type in the composer (Enter sends, Shift+Enter makes a new line).
//
// One turn, as the learner sees it:
//   listening  -> the mic button is held (or hands-free hears speech)
//   thinking   -> recording sent; waiting for the transcript and first sentence
//   speaking   -> coach audio is playing, sentence by sentence
//   idle       -> the learner's turn
//
// The server streams NDJSON events per turn (see coach/voice.py):
//   transcript -> fills the learner's bubble with what Whisper heard
//   text       -> grows the coach's bubble token by token
//   audio      -> queued in CoachPlayer, played in order
//   done / error / aborted
//
// Help that never interrupts the conversation:
//   - feedback markers under the learner's OWN messages, as the learner chose
//     (Settings or the switch in the header): after the session / only with
//     Sam / gentle / every mistake. They come from the background analyzers,
//     so they appear a moment after the reply. Characters never correct.
//   - "Need an idea?": three things the learner could say next
//   - "Say it better": a fluent version of one of their messages
//   - select a word in the coach's reply to see what it means

import { api, readEvents, sendJSON, getJSON } from "./api.js";
import { Mic, CoachPlayer, audioContext, MIC_SVG } from "./audio.js";
import { EnergyVAD } from "./vad.js";
import { renderMarkdown } from "./markdown.js";
import { showFindings } from "./feedback.js";
import { esc, toast, state, getStatus, refreshSidebar, autoGrow, FEEDBACK_INFO } from "./app.js";

const MIN_RECORDING_MS = 350;   // shorter presses are accidental taps, not speech
const PTT_HINT = "Hold the mic (or Space) while you speak, or tap once to start and again to stop.";
const FEEDBACK_POLL_MS = 900;
const norm = (s) => s.replace(/\s+/g, " ").trim().toLowerCase();

// opts: { mode: "tutor", firstMessage?, resumeId? } or { mode: "roleplay", personaId }
export async function startTalk(view, opts) {
  const status = getStatus() || {};
  const voiceIn = status.stt?.state === "ready";
  const voiceOut = status.tts?.state === "ready";

  view.innerHTML = `<div class="empty"><span class="spinner"></span><p>Getting the room ready...</p></div>`;
  audioContext();   // we got here from a click, so audio playback is allowed to start
  const body = opts.resumeId ? { resume_id: opts.resumeId }
    : opts.mode === "tutor" ? { mode: "tutor" } : { persona_id: opts.personaId };
  const start = await sendJSON("/api/sessions", body);
  const sid = start.id;
  const persona = start.persona;
  const tutor = start.mode === "tutor";
  const initial = persona.name.replace(/^(Mr|Ms|Mrs|Dr|Officer)\.?\s+/, "")[0];   // "Mr. Rao" -> "R"

  view.innerHTML = `
    <section class="talk ${tutor ? "is-tutor" : "is-roleplay"}">
      <div class="talk-head">
        <div class="avatar" aria-hidden="true">${esc(initial)}</div>
        <div class="grow"><h2>${esc(persona.name)} <span class="muted small">${esc(tutor ? "English tutor" : persona.role)}</span></h2>
          <div class="state" id="state" data-state="idle"><i class="dot"></i><span>Your turn</span></div></div>
        <label class="fb-switch" title="Feedback during chats (you can also change it in Settings)">
          <span class="small muted">Feedback</span>
          <select id="fbmode">${Object.entries(FEEDBACK_INFO).map(([v, [t]]) => `<option value="${v}">${esc(t)}</option>`).join("")}</select>
        </label>
        <button class="danger" id="end">${tutor ? "Finish" : "End session"}</button>
      </div>
      <div class="transcript" id="log" aria-live="polite"></div>
      <div class="controls">
        ${voiceIn ? "" : `<div class="banner small">${esc(status.stt?.message || "Voice input is off.")} You can type instead.</div>`}
        ${voiceOut ? "" : `<div class="banner small">${esc(status.tts?.message || "Coach voice is off.")} Replies appear as text.</div>`}
        <div class="hints" id="hints" hidden></div>
        <div class="tools">
          <button class="chip" id="hint-btn" type="button">Need an idea?</button>
          ${voiceIn ? `<label class="switch"><input type="checkbox" id="handsfree"> Hands-free</label>` : ""}
          <span class="level-meter" aria-hidden="true"><i id="meter"></i></span>
        </div>
        <div class="composer-row">
          <form class="composer" id="typebox">
            <textarea id="typed" rows="1" placeholder="${tutor ? "Ask Sam anything, or just talk…" : "Type your answer…"}" aria-label="Type your message"></textarea>
            <button class="primary" type="submit">Send</button>
          </form>
          <button class="mic" id="mic" aria-label="Hold to talk" ${voiceIn ? "" : "disabled"}>${MIC_SVG}</button>
        </div>
        <div class="mic-hint" id="hint">${voiceIn ? PTT_HINT : ""}</div>
      </div>
    </section>`;

  const $ = (id) => view.querySelector(`#${id}`);
  const log = $("log"), micBtn = $("mic"), hint = $("hint"), stateEl = $("state"), typed = $("typed");
  const mic = new Mic();
  let busy = false;            // a turn is in flight (thinking or speaking)
  let ended = false;
  let fetchCtl = null;         // aborts the streaming request on barge-in
  let coachBubble = null;
  let pressStart = 0;
  let tapMode = false;         // a quick tap starts recording; the next tap stops it
  let meterTimer = null;
  let vad = null;              // set while hands-free mode is on
  let pollTimer = null;
  const mine = [];             // the learner's message bubbles, in order (feedback attaches here)

  // ---- UI state ----
  const LABEL = { idle: "Your turn", listening: "Listening...", thinking: "Thinking...", speaking: `${persona.name} is speaking` };
  function setState(s) {
    stateEl.dataset.state = s;
    stateEl.querySelector("span").textContent = LABEL[s];
    busy = s === "thinking" || s === "speaking";
    // Hands-free: don't listen while waiting for the reply (a "hmm" during the
    // wait would become a new turn), but DO listen while the coach speaks, so
    // the learner can interrupt it.
    if (vad) s === "thinking" ? vad.pause() : vad.resume();
  }
  const scrollDown = () => { log.scrollTop = log.scrollHeight; };

  // ---- message bubbles ----
  function coachMessage(text = "") {
    const wrap = document.createElement("div");
    wrap.className = "msg coach";
    wrap.innerHTML = `<div class="bubble coach"></div>
      <div class="msg-actions">${voiceOut ? `<button class="link" data-act="replay">Play again</button>` : ""}</div>`;
    const b = wrap.querySelector(".bubble");
    b._text = "";
    b._wavs = [];
    log.appendChild(wrap);
    setCoachText(b, text);
    scrollDown();
    return b;
  }
  function setCoachText(b, text) {
    b._text = text;
    // The tutor formats explanations (bold, lists); characters just talk.
    if (tutor) b.innerHTML = renderMarkdown(text);
    else b.textContent = text;
  }
  function myMessage(text, pending = false) {
    const wrap = document.createElement("div");
    wrap.className = "msg me";
    wrap.innerHTML = `<div class="bubble me ${pending ? "pending" : ""}"></div><div class="fb" hidden></div>`;
    wrap.querySelector(".bubble").textContent = text;
    log.appendChild(wrap);
    mine.push(wrap);
    scrollDown();
    return wrap;
  }
  const note = (text) => {
    const n = document.createElement("div");
    n.className = "bubble note";
    n.textContent = text;
    log.appendChild(n);
    scrollDown();
  };

  const player = new CoachPlayer({
    onStart: () => setState("speaking"),
    onIdle: () => { if (!fetchCtl) setState("idle"); },
  });

  // ---- opening line, or the history of a resumed chat ----
  if (start.opening) {
    const b = coachMessage(start.opening.text);
    if (start.opening.wav) {
      b._wavs = [start.opening.wav];
      player.reset();
      player.enqueue(0, start.opening.wav).then(() => player.end());
    }
  }
  for (const m of start.history || []) {
    if (m.role === "assistant") coachMessage(m.content);
    else myMessage(m.content);
  }
  if ((start.history || []).length) note("Welcome back! Carry on where you left off.");

  // ---- feedback markers (the learner's choice) ----
  const policy = () => state.profile?.chat_feedback || "after";
  const feedbackOn = () => ["gentle", "all"].includes(policy()) || (policy() === "tutor_only" && tutor);
  const betterOn = () => tutor || ["gentle", "all"].includes(policy());
  $("fbmode").value = policy();

  function renderFeedback(wrap, findings) {
    const fb = wrap.querySelector(".fb");
    const text = wrap.querySelector(".bubble").textContent;
    fb.hidden = false;
    wrap._hasFeedback = true;
    const better = betterOn() ? ` <button class="link" data-act="better">Say it better</button>` : "";
    if (!findings.length) {
      fb.innerHTML = `<span class="fb-ok">✓ Sounds good</span>${better}`;
      return;
    }
    fb.innerHTML = `<button class="fb-chip" data-act="toggle" aria-expanded="false">${findings.length} tip${findings.length > 1 ? "s" : ""}</button>${better}
      <div class="fb-panel" hidden><p class="sentence fb-sentence"></p>
        <p class="muted small">Tap a highlight to see why and what to try instead.</p></div>`;
    showFindings(fb.querySelector(".fb-sentence"), text, findings);
  }

  function clearFeedback() {
    mine.forEach((w) => { w._hasFeedback = false; const fb = w.querySelector(".fb"); fb.hidden = true; fb.innerHTML = ""; });
    log.querySelectorAll(".better").forEach((b) => b.remove());
  }

  // Poll until every message has its feedback (analysis runs after each reply).
  async function pollFeedback(tries = 40) {
    clearTimeout(pollTimer);
    if (ended || !feedbackOn()) return;
    let turns = [];
    try {
      turns = (await getJSON(`/api/sessions/${sid}/feedback`)).turns;
    } catch { return; }
    // Match by text: a turn's text is exactly what the learner's bubble shows.
    const open = mine.filter((w) => !w._hasFeedback);
    for (const t of turns) {
      const i = open.findIndex((w) => norm(w.querySelector(".bubble").textContent) === norm(t.text));
      if (i >= 0) renderFeedback(open.splice(i, 1)[0], t.findings);
    }
    const waiting = mine.some((w) => !w._hasFeedback && !w.querySelector(".bubble").classList.contains("pending"));
    if (waiting && tries > 0) pollTimer = setTimeout(() => pollFeedback(tries - 1), FEEDBACK_POLL_MS);
  }

  $("fbmode").addEventListener("change", async (e) => {
    try {
      state.profile = (await sendJSON("/api/profile", { ...state.profile, chat_feedback: e.target.value }, "PUT")).profile;
    } catch (err) { toast(err.message); return; }
    clearFeedback();
    if (feedbackOn()) pollFeedback(); else toast("Feedback will be in your report after the session.");
  });

  // ---- clicks inside the conversation ----
  log.addEventListener("click", async (e) => {
    const btn = e.target.closest("button[data-act]");
    if (!btn) return;
    const wrap = btn.closest(".msg");
    if (btn.dataset.act === "toggle") {
      const panel = wrap.querySelector(".fb-panel");
      panel.hidden = !panel.hidden;
      btn.setAttribute("aria-expanded", String(!panel.hidden));
    } else if (btn.dataset.act === "better") {
      btn.disabled = true;
      btn.textContent = "Thinking…";
      try {
        const r = await sendJSON(`/api/sessions/${sid}/better`, { text: wrap.querySelector(".bubble").textContent });
        const box = document.createElement("div");
        box.className = "better";
        box.innerHTML = `<b>Better:</b> ${esc(r.better)} ${voiceOut ? `<button class="link" data-act="say">Hear it</button>` : ""}
          <div class="muted small">${esc(r.why)}</div>`;
        box.querySelector("[data-act=say]")?.addEventListener("click", (ev) => { ev.stopPropagation(); speak(r.better); });
        wrap.appendChild(box);
        btn.remove();
        scrollDown();
      } catch (err) { toast(err.message); btn.disabled = false; btn.textContent = "Say it better"; }
    } else if (btn.dataset.act === "replay") {
      const b = wrap.querySelector(".bubble");
      if (busy) await bargeIn();
      if (b._wavs.length) {
        player.reset();
        b._wavs.forEach((w, i) => player.enqueue(i, w));
        player.end();
      } else {
        speak(b._text);   // e.g. a message from an earlier visit to this chat
      }
    }
  });

  async function speak(text) {
    try {
      const { wav } = await sendJSON("/api/speak", { text, session_id: sid });
      player.reset();
      await player.enqueue(0, wav);
      player.end();
    } catch (err) { toast(err.message); }
  }

  // ---- select a word in the coach's reply to see its meaning ----
  const lookup = document.createElement("button");
  lookup.className = "lookup-btn";
  lookup.hidden = true;
  document.body.appendChild(lookup);
  const onSelect = () => {
    const sel = window.getSelection();
    const word = sel?.toString().trim() || "";
    const bubble = sel?.anchorNode?.parentElement?.closest(".bubble.coach");
    if (!bubble || !word || word.split(/\s+/).length > 3 || word.length > 40) { lookup.hidden = true; return; }
    const r = sel.getRangeAt(0).getBoundingClientRect();
    lookup.textContent = `Meaning of “${word}”`;
    lookup.style.left = `${window.scrollX + r.left}px`;
    lookup.style.top = `${window.scrollY + r.bottom + 6}px`;
    lookup.hidden = false;
    lookup.onclick = async (ev) => {
      ev.stopPropagation();
      lookup.textContent = "Looking it up…";
      try {
        const d = await sendJSON("/api/define", { word, sentence: bubble._text });
        const box = document.createElement("div");
        box.className = "define";
        box.innerHTML = `<b>${esc(d.word)}</b> <span class="muted small">${esc(d.part_of_speech)}</span>
          <div>${esc(d.meaning)}</div><div class="muted small">e.g. ${esc(d.example)}</div>`;
        bubble.closest(".msg").appendChild(box);
        scrollDown();
      } catch (err) { toast(err.message); }
      lookup.hidden = true;
    };
  };
  log.addEventListener("mouseup", () => setTimeout(onSelect, 0));
  log.addEventListener("touchend", () => setTimeout(onSelect, 300));
  const hideLookup = (e) => { if (e.target !== lookup) lookup.hidden = true; };
  document.addEventListener("mousedown", hideLookup);

  // ---- "Need an idea?" ----
  $("hint-btn").addEventListener("click", async () => {
    const box = $("hints");
    const btn = $("hint-btn");
    btn.disabled = true;
    btn.textContent = "Thinking of ideas…";
    try {
      const { suggestions } = await sendJSON(`/api/sessions/${sid}/hint`, {});
      box.innerHTML = `<span class="muted small">You could say:</span>` +
        suggestions.map((s) => `<button class="chip" type="button">${esc(s)}</button>`).join("");
      box.hidden = false;
    } catch (err) { toast(err.message); }
    btn.disabled = false;
    btn.textContent = "Need an idea?";
  });
  $("hints").addEventListener("click", (e) => {
    const c = e.target.closest("button.chip");
    if (!c) return;
    typed.value = c.textContent;   // say it out loud, or send it as typed
    typed.dispatchEvent(new Event("input"));
    typed.focus();
  });

  // ---- one turn ----
  async function sendTurn({ blob = null, text = null }) {
    if (ended) return;
    if (busy) await bargeIn();
    $("hints").hidden = true;
    const form = new FormData();
    let me;
    if (blob) {
      form.append("audio", blob, "turn.webm");
      me = myMessage("...", true);
    } else {
      form.append("text", text);
      me = myMessage(text);
    }
    const myBubble = me.querySelector(".bubble");
    setState("thinking");
    player.reset();
    coachBubble = null;
    let gotAudio = false;
    const ctl = (fetchCtl = new AbortController());
    try {
      const res = await api(`/api/sessions/${sid}/turn`, { method: "POST", form, signal: ctl.signal });
      await readEvents(res, (ev) => {
        if (ctl !== fetchCtl) return;          // a newer turn took over
        if (ev.type === "transcript") {
          if (ev.text.trim()) { myBubble.textContent = ev.text; myBubble.classList.remove("pending"); }
          else { me.remove(); mine.splice(mine.indexOf(me), 1); note("I didn't catch that. Try again, a little closer to the mic."); }
        } else if (ev.type === "text") {
          if (!coachBubble) coachBubble = coachMessage("");
          setCoachText(coachBubble, coachBubble._text + ev.delta);
          scrollDown();
        } else if (ev.type === "audio" && ev.wav) {
          gotAudio = true;
          coachBubble?._wavs.push(ev.wav);
          player.enqueue(ev.index, ev.wav);
        } else if (ev.type === "error") {
          myBubble.classList.remove("pending");
          note(ev.message);
        }
      });
    } catch (err) {
      if (err.name !== "AbortError") note(err.message);
    } finally {
      // Tidy the finished reply: roleplay characters sometimes leave blank lines.
      if (coachBubble && !tutor) setCoachText(coachBubble, coachBubble._text.replace(/\s*\n\s*/g, " ").trim());
      if (ctl === fetchCtl) {
        fetchCtl = null;
        if (gotAudio) player.end(); else { player.stop(); setState("idle"); }
      }
      pollFeedback();
    }
  }

  // Stop the coach now: silence the audio, drop the stream, tell the server.
  async function bargeIn() {
    player.stop();
    if (fetchCtl) {
      fetchCtl.abort();
      fetchCtl = null;
      if (coachBubble) setCoachText(coachBubble, coachBubble._text + " …");
      try { await sendJSON(`/api/sessions/${sid}/abort`, {}); } catch { /* session may have ended */ }
    }
    setState("idle");
  }

  // ---- push-to-talk ----
  async function beginRecording() {
    if (ended || mic.recording || vad) return;
    try {
      await mic.open();
    } catch (err) {
      toast(err.name === "NotAllowedError" ? "Microphone permission was denied." : err.message);
      tapMode = false;
      return;
    }
    if (busy) await bargeIn();    // speaking over the coach interrupts it
    mic.start();
    micBtn.classList.add("recording");
    setState("listening");
    meterTimer = setInterval(() => { $("meter").style.width = `${Math.min(100, mic.level() * 600)}%`; }, 60);
  }

  async function finishRecording() {
    if (!mic.recording) { tapMode = false; return; }
    const ms = performance.now() - mic.startedAt;
    clearInterval(meterTimer);
    $("meter").style.width = "0";
    micBtn.classList.remove("recording");
    tapMode = false;
    hint.textContent = PTT_HINT;
    const blob = await mic.stop();
    if (!blob || ms < MIN_RECORDING_MS) { setState("idle"); return; }
    sendTurn({ blob });
  }

  if (voiceIn) {
    micBtn.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      micBtn.setPointerCapture?.(e.pointerId);
      if (tapMode) { finishRecording(); return; }
      pressStart = performance.now();
      beginRecording();
    });
    micBtn.addEventListener("pointerup", () => {
      if (tapMode) return;
      // A quick tap, or a release while the mic was still opening (first-time
      // permission prompt): keep recording until the next tap.
      if (performance.now() - pressStart < MIN_RECORDING_MS || !mic.recording) {
        tapMode = true;
        hint.textContent = "Recording... tap again when you're done.";
        return;
      }
      finishRecording();
    });
    micBtn.addEventListener("contextmenu", (e) => e.preventDefault());   // long-press menu on phones
  }

  // Space bar = push-to-talk, unless the learner is typing.
  const onKey = (e) => {
    if (!voiceIn || e.code !== "Space" || e.target.closest("input, textarea, select")) return;
    e.preventDefault();
    if (e.type === "keydown" && !e.repeat) beginRecording();
    if (e.type === "keyup") finishRecording();
  };
  document.addEventListener("keydown", onKey);
  document.addEventListener("keyup", onKey);

  // ---- hands-free ----
  async function startHandsFree() {
    try {
      await mic.open();
    } catch (err) {
      toast(err.name === "NotAllowedError" ? "Microphone permission was denied." : err.message);
      $("handsfree").checked = false;
      return;
    }
    const silenceMs = state.profile?.silence_ms || 1200;
    vad = new EnergyVAD(mic, {
      silenceMs,
      isCoachSpeaking: () => player.playing,
      onLevel: (lvl) => { $("meter").style.width = `${Math.min(100, lvl * 600)}%`; },
      onSpeechStart: async () => {
        if (busy) await bargeIn();          // the learner talked over the coach
        micBtn.classList.add("recording");
        setState("listening");
      },
      onSpeechEnd: (blob) => {
        micBtn.classList.remove("recording");
        if (blob) sendTurn({ blob }); else setState("idle");
      },
    });
    vad.start();
    micBtn.disabled = true;
    hint.textContent = `Hands-free: just speak. A ${(silenceMs / 1000).toFixed(1)} s pause ends your turn; talk over ${persona.name} to interrupt.`;
    if (busy) vad.resume(); else setState("idle");
  }

  function stopHandsFree() {
    vad?.stop();
    vad = null;
    micBtn.disabled = false;
    micBtn.classList.remove("recording");
    $("meter").style.width = "0";
    hint.textContent = PTT_HINT;
    if (!busy) setState("idle");
  }

  $("handsfree")?.addEventListener("change", (e) => (e.target.checked ? startHandsFree() : stopHandsFree()));

  // ---- typed turns ----
  autoGrow(typed);
  const sendTyped = () => {
    const text = typed.value.trim();
    if (!text) return;
    typed.value = "";
    typed.dispatchEvent(new Event("input"));
    sendTurn({ text });
  };
  typed.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendTyped(); }
  });
  $("typebox").addEventListener("submit", (e) => { e.preventDefault(); sendTyped(); });

  // ---- ending ----
  const onPageHide = () => { if (!ended) navigator.sendBeacon(`/api/sessions/${sid}/end`); };
  window.addEventListener("pagehide", onPageHide);   // closing the browser tab still saves the chat

  async function endSession() {
    if (ended) return null;
    ended = true;
    clearInterval(meterTimer);
    clearTimeout(pollTimer);
    vad?.stop();
    vad = null;
    document.removeEventListener("keydown", onKey);
    document.removeEventListener("keyup", onKey);
    document.removeEventListener("mousedown", hideLookup);
    window.removeEventListener("pagehide", onPageHide);
    lookup.remove();
    player.stop();
    fetchCtl?.abort();
    mic.close();
    const r = await sendJSON(`/api/sessions/${sid}/end`, {});
    refreshSidebar();
    return r;
  }

  $("end").addEventListener("click", async () => {
    $("end").disabled = true;
    view.querySelector(".controls").innerHTML = `<div class="empty"><span class="spinner"></span><p>Preparing your feedback...</p></div>`;
    try {
      const report = await endSession();
      if (report.empty) { toast("No turns yet, so there's nothing to report."); location.hash = "#/home"; return; }
      state.reportCache.set(String(report.id), report);
      location.hash = `#/report/${report.id}`;
    } catch (err) {
      toast(err.message);
      location.hash = "#/home";
    }
  });

  // Navigating away mid-conversation still saves it (the learner's turns count).
  state.leave = async () => {
    if (ended) return;
    try {
      const r = await endSession();
      if (r && !r.empty) toast(tutor ? "Chat saved. Find it in the sidebar." : "Session saved. Find it under Progress.");
    } catch { /* already gone */ }
  };

  // A message typed on the start page goes first.
  if (opts.firstMessage) sendTurn({ text: opts.firstMessage });
  else if (feedbackOn() && mine.length) pollFeedback();
  typed.focus({ preventScroll: true });
}
