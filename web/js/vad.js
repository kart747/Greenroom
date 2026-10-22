// Hands-free mode: an energy-based voice activity detector (VAD) in the browser.
//
// How it works: every 30 ms we read the microphone's loudness (RMS energy, from
// the AnalyserNode in audio.js) and compare it with a threshold.
//   - speech starts when the level stays above the threshold for ~150 ms
//     (shorter blips are a cough, a click or a door, not speech);
//   - the turn ends after `silenceMs` of quiet (default 1200 ms). Learners
//     often pause mid-sentence to find a word, so a short gap must NOT end
//     their turn; 1.2 s is long enough for that and still feels responsive.
//
// Why energy and not Silero VAD (a neural VAD) in the browser:
//   + no extra model to download or ship, nothing to load: works offline at once
//   + a few lines of code, cheap enough to run every 30 ms
//   - it only measures loudness, so in a noisy room (fan, traffic, other people
//     talking) it can mistake noise for speech or miss quiet speech. The
//     adaptive noise floor below helps with steady noise, not with voices.
//   (The server still runs Silero inside faster-whisper, via vad_filter=True,
//   to drop non-speech before transcription.)
//
// Recording starts on the FIRST loud frame, not after the 150 ms check, so the
// first syllable is not cut off. If the sound turns out to be a blip, that
// tentative recording is thrown away.

const FRAME_MS = 30;
const MIN_THRESHOLD = 0.012;    // never trigger below this (a very quiet room)
const NOISE_MULTIPLIER = 3;     // speech must be ~3x louder than the background
const KEEP_GOING = 0.6;         // once speaking, a softer level still counts (hysteresis)
const MAX_TURN_MS = 30000;      // safety cap on one turn

export class EnergyVAD {
  constructor(mic, { startMs = 150, silenceMs = 1200, onSpeechStart, onSpeechEnd, onLevel, isCoachSpeaking } = {}) {
    this.mic = mic;
    this.startMs = startMs;
    this.silenceMs = silenceMs;
    this.onSpeechStart = onSpeechStart || (() => {});
    this.onSpeechEnd = onSpeechEnd || (() => {});
    this.onLevel = onLevel || (() => {});
    // While the coach talks, some of its voice can leak past echo cancellation;
    // a higher bar then stops the coach from "interrupting itself".
    this.isCoachSpeaking = isCoachSpeaking || (() => false);
    this.noise = 0.01;          // running estimate of background loudness
    this.phase = "quiet";       // quiet -> maybe (tentative) -> speech
    this.loudSince = 0;
    this.lastLoud = 0;
    this.timer = null;
    this.paused = false;
  }

  threshold() {
    const t = Math.max(MIN_THRESHOLD, this.noise * NOISE_MULTIPLIER);
    return this.isCoachSpeaking() ? t * 2 : t;
  }

  start() {
    if (this.timer) return;
    this.phase = "quiet";
    this.timer = setInterval(() => this._tick(), FRAME_MS);
  }

  stop() {
    clearInterval(this.timer);
    this.timer = null;
    if (this.phase !== "quiet") this.mic.stop();   // drop any half-recorded turn
    this.phase = "quiet";
  }

  // Pause while a turn is being processed, so the VAD doesn't record the
  // learner's "hmm" during the wait as a new turn. Barge-in still works
  // because the VAD resumes as soon as the coach starts speaking.
  pause() { this.paused = true; }
  resume() { this.paused = false; }

  _tick() {
    const level = this.mic.level();
    const now = performance.now();
    this.onLevel(level);
    if (this.paused) return;
    const thr = this.threshold();

    if (this.phase === "quiet") {
      if (level > thr) {
        this.phase = "maybe";
        this.loudSince = now;
        this.lastLoud = now;
        this.mic.start();                       // start early: keep the first syllable
      } else {
        // Slowly follow the background level (fan noise, room hum).
        this.noise = 0.97 * this.noise + 0.03 * level;
      }
      return;
    }

    if (level > thr * KEEP_GOING) this.lastLoud = now;

    if (this.phase === "maybe") {
      if (now - this.lastLoud > 2 * FRAME_MS) {   // went quiet again: it was a blip
        this.phase = "quiet";
        this.mic.stop();                           // discard the tentative recording
      } else if (now - this.loudSince >= this.startMs) {
        this.phase = "speech";
        this.onSpeechStart();
      }
      return;
    }

    // phase === "speech"
    if (now - this.lastLoud >= this.silenceMs || now - this.loudSince > MAX_TURN_MS) {
      this.phase = "quiet";
      this.mic.stop().then((blob) => this.onSpeechEnd(blob));
    }
  }
}
