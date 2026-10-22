// Microphone recording and coach audio playback, in the browser.
//
// Mic: opened ONCE per conversation (getUserMedia), then MediaRecorder records
// each turn as webm/opus. Keeping the stream open avoids a permission prompt
// and ~0.5 s of mic start-up on every turn.
// echoCancellation is on so the coach's voice from the speakers is not picked
// up as the learner speaking (this matters for hands-free mode and barge-in).
//
// Playback: the server sends one WAV per sentence. They can arrive faster than
// they play, so CoachPlayer queues them and plays them back to back, in order.

let ctx = null;

// One shared AudioContext. Browsers only allow it to start after a click,
// so it is created/resumed from a user action (the "start" or mic button).
export function audioContext() {
  if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
  if (ctx.state === "suspended") ctx.resume();
  return ctx;
}

export class Mic {
  constructor() {
    this.stream = null;
    this.analyser = null;
    this.recorder = null;
    this.chunks = [];
    this._buf = null;
  }

  async open() {
    if (this.stream) return;
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("This window can't use the microphone. Type your answers instead.");
    }
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
    });
    // The analyser gives the live input level: used for the level meter and
    // for the hands-free voice-activity detector.
    const ac = audioContext();
    const src = ac.createMediaStreamSource(this.stream);
    this.analyser = ac.createAnalyser();
    this.analyser.fftSize = 1024;
    src.connect(this.analyser);
    this._buf = new Float32Array(this.analyser.fftSize);
  }

  // Root-mean-square energy of the latest ~20 ms of input: 0 (silence) .. ~0.5 (shouting).
  level() {
    if (!this.analyser) return 0;
    this.analyser.getFloatTimeDomainData(this._buf);
    let sum = 0;
    for (const v of this._buf) sum += v * v;
    return Math.sqrt(sum / this._buf.length);
  }

  get recording() { return this.recorder?.state === "recording"; }

  start() {
    if (this.recording) return;
    this.chunks = [];
    const type = MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "";
    this.recorder = new MediaRecorder(this.stream, type ? { mimeType: type } : undefined);
    this.recorder.ondataavailable = (e) => { if (e.data.size) this.chunks.push(e.data); };
    this.recorder.start();
    this.startedAt = performance.now();
  }

  // Stop and return the recording as a Blob (null if nothing was recorded).
  stop() {
    return new Promise((resolve) => {
      if (!this.recording) return resolve(null);
      this.recorder.onstop = () => {
        const blob = this.chunks.length ? new Blob(this.chunks, { type: this.recorder.mimeType || "audio/webm" }) : null;
        resolve(blob);
      };
      this.recorder.stop();
    });
  }

  close() {
    if (this.recording) this.recorder.stop();
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.analyser = null;
  }
}

// Plays the coach's sentences in order, as they arrive.
export class CoachPlayer {
  constructor({ onStart, onIdle } = {}) {
    this.onStart = onStart || (() => {});
    this.onIdle = onIdle || (() => {});
    this.pending = new Map();   // index -> decoded AudioBuffer, waiting for its turn
    this.next = 0;              // index of the sentence that should play next
    this.source = null;         // the sentence playing right now
    this.generation = 0;        // bumped by stop(): late decodes from an old reply are ignored
    this.finished = true;       // has the server sent every sentence of this reply?
  }

  get playing() { return this.source !== null; }

  // A new reply begins: sentence numbering restarts at 0.
  reset() {
    this.stop();
    this.finished = false;
  }

  // The server said the reply is complete; when the queue drains we are idle.
  end() {
    this.finished = true;
    if (!this.playing && this.pending.size === 0) this.onIdle();
  }

  async enqueue(index, base64wav) {
    const gen = this.generation;
    const bytes = Uint8Array.from(atob(base64wav), (c) => c.charCodeAt(0));
    const buffer = await audioContext().decodeAudioData(bytes.buffer);
    if (gen !== this.generation) return;      // stop() happened while decoding
    this.pending.set(index, buffer);
    this._pump();
  }

  _pump() {
    if (this.playing || !this.pending.has(this.next)) return;
    const buffer = this.pending.get(this.next);
    this.pending.delete(this.next);
    this.next += 1;
    const ac = audioContext();
    const src = ac.createBufferSource();
    src.buffer = buffer;
    src.connect(ac.destination);
    src.onended = () => {
      if (this.source !== src) return;        // stopped on purpose
      this.source = null;
      if (this.pending.has(this.next)) this._pump();
      else if (this.finished) this.onIdle();
    };
    this.source = src;
    this.onStart();
    src.start();
  }

  // Barge-in or leaving the screen: silence now and forget the queue.
  stop() {
    this.generation += 1;
    this.pending.clear();
    this.next = 0;
    if (this.source) {
      const s = this.source;
      this.source = null;
      try { s.stop(); } catch { /* already ended */ }
    }
  }
}
