// The conversation screen.
//
// Two ways to talk:
//   push-to-talk  hold the mic button (or Space); release to send
//   hands-free    toggle on; the voice-activity detector in vad.js starts and
//                 stops recording by itself, and speaking while the coach
//                 talks interrupts it (barge-in)
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

import { api, readEvents, sendJSON } from "./api.js";
import { Mic, CoachPlayer, audioContext } from "./audio.js";
import { EnergyVAD } from "./vad.js";
import { esc, toast, state, getStatus } from "./app.js";

const MIC_SVG = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true">
  <rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/></svg>`;
const MIN_RECORDING_MS = 350;   // shorter presses are accidental taps, not speech
const PTT_HINT = "Hold the button (or Space) while you speak, or tap once to start and again to stop.";

export async function startTalk(view, personaId) {
  const status = getStatus() || {};
  const voiceIn = status.stt?.state === "ready";
  const voiceOut = status.tts?.state === "ready";

  view.innerHTML = `<div class="empty"><span class="spinner"></span><p>Getting the room ready...</p></div>`;
  audioContext();   // we got here from a click, so audio playback is allowed to start
  const start = await sendJSON("/api/sessions", { persona_id: personaId });
  const sid = start.id;
  const persona = start.persona;

  view.innerHTML = `
    <section class="talk">
      <div class="talk-head">
        <div class="avatar" aria-hidden="true">${esc(persona.name.replace(/^(Mr|Ms|Mrs|Dr|Officer)\.?\s+/, "")[0])}</div>
        <div class="grow"><h2 style="margin:0">${esc(persona.name)}</h2>
          <div class="state" id="state" data-state="idle"><i class="dot"></i><span>Your turn</span></div></div>
        <button class="danger" id="end">End session</button>
      </div>
      <div class="transcript" id="log" aria-live="polite"></div>
      <div class="controls">
        ${voiceIn ? "" : `<div class="banner small">${esc(status.stt?.message || "Voice input is off.")} You can type instead.</div>`}
        ${voiceOut ? "" : `<div class="banner small">${esc(status.tts?.message || "Coach voice is off.")} Replies appear as text.</div>`}
        <div class="mic-row">
          <span class="level-meter" aria-hidden="true"><i id="meter"></i></span>
          <button class="mic" id="mic" aria-label="Hold to talk" ${voiceIn ? "" : "disabled"}>${MIC_SVG}</button>
          <span style="width:140px"></span>
        </div>
        <div class="mic-hint" id="hint">${voiceIn ? PTT_HINT : ""}</div>
        ${voiceIn ? `<label class="switch" style="justify-self:center"><input type="checkbox" id="handsfree"> Hands-free (just talk, pause to finish)</label>` : ""}
        <form class="typebox" id="typebox"><input type="text" id="typed" placeholder="Or type your answer..." autocomplete="off" aria-label="Type your answer">
          <button type="submit">Send</button></form>
      </div>
    </section>`;

  const $ = (id) => view.querySelector(`#${id}`);
  const log = $("log"), micBtn = $("mic"), hint = $("hint"), stateEl = $("state");
  const mic = new Mic();
  let busy = false;            // a turn is in flight (thinking or speaking)
  let ended = false;
  let fetchCtl = null;         // aborts the streaming request on barge-in
  let coachBubble = null;
  let pressStart = 0;
  let tapMode = false;         // a quick tap starts recording; the next tap stops it
  let meterTimer = null;
  let vad = null;              // set while hands-free mode is on

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
  function bubble(cls, text) {
    const b = document.createElement("div");
    b.className = `bubble ${cls}`;
    b.textContent = text;
    log.appendChild(b);
    log.scrollTop = log.scrollHeight;
    return b;
  }
  const note = (text) => bubble("note", text);

  const player = new CoachPlayer({
    onStart: () => setState("speaking"),
    onIdle: () => { if (!fetchCtl) setState("idle"); },
  });

  // ---- opening line ----
  bubble("coach", start.opening.text);
  if (start.opening.wav) {
    player.reset();
    player.enqueue(0, start.opening.wav).then(() => player.end());
  }

  // ---- one turn ----
  async function sendTurn({ blob = null, text = null }) {
    if (ended) return;
    if (busy) await bargeIn();
    const form = new FormData();
    let mine;
    if (blob) {
      form.append("audio", blob, "turn.webm");
      mine = bubble("me pending", "...");
    } else {
      form.append("text", text);
      mine = bubble("me", text);
    }
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
          if (ev.text.trim()) { mine.textContent = ev.text; mine.classList.remove("pending"); }
          else { mine.remove(); note("I didn't catch that. Try again, a little closer to the mic."); }
        } else if (ev.type === "text") {
          if (!coachBubble) coachBubble = bubble("coach", "");
          coachBubble.textContent += ev.delta;
          log.scrollTop = log.scrollHeight;
        } else if (ev.type === "audio" && ev.wav) {
          gotAudio = true;
          player.enqueue(ev.index, ev.wav);
        } else if (ev.type === "error") {
          mine.classList.remove("pending");
          note(ev.message);
        }
      });
    } catch (err) {
      if (err.name !== "AbortError") note(err.message);
    } finally {
      // Tidy the finished reply: the LLM sometimes leaves blank lines between sentences.
      if (coachBubble) coachBubble.textContent = coachBubble.textContent.replace(/\s*\n\s*/g, " ").trim();
      if (ctl === fetchCtl) {
        fetchCtl = null;
        if (gotAudio) player.end(); else { player.stop(); setState("idle"); }
      }
    }
  }

  // Stop the coach now: silence the audio, drop the stream, tell the server.
  async function bargeIn() {
    player.stop();
    if (fetchCtl) {
      fetchCtl.abort();
      fetchCtl = null;
      if (coachBubble) coachBubble.textContent += " …";
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
    hint.textContent = `Hands-free: just speak. A ${(silenceMs / 1000).toFixed(1)} s pause ends your turn; talk over the coach to interrupt.`;
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
  $("typebox").addEventListener("submit", (e) => {
    e.preventDefault();
    const input = $("typed");
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    sendTurn({ text });
  });

  // ---- ending ----
  async function endSession() {
    if (ended) return null;
    ended = true;
    clearInterval(meterTimer);
    vad?.stop();
    vad = null;
    document.removeEventListener("keydown", onKey);
    document.removeEventListener("keyup", onKey);
    player.stop();
    fetchCtl?.abort();
    mic.close();
    return sendJSON(`/api/sessions/${sid}/end`, {});
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
      if (r && !r.empty) toast("Session saved. Find it under Progress.");
    } catch { /* already gone */ }
  };
}
