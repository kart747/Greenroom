// Unit tests for web/js/vad.js with a fake mic and a fake clock (run by tests/test_web_js.py).
import assert from "node:assert/strict";
import { EnergyVAD } from "../../web/js/vad.js";

let now = 0;
globalThis.performance = { now: () => now };

class FakeMic {
  constructor() { this.lvl = 0; this.recording = false; this.starts = 0; this.stops = 0; }
  level() { return this.lvl; }
  start() { this.recording = true; this.starts++; }
  stop() { this.recording = false; this.stops++; return Promise.resolve("blob"); }
}

// Feed `ms` of audio at a given level, one 30 ms frame at a time.
function feed(vad, mic, level, ms) {
  for (let t = 0; t < ms; t += 30) { mic.lvl = level; now += 30; vad._tick(); }
}

function make(opts = {}) {
  const mic = new FakeMic();
  const events = [];
  const vad = new EnergyVAD(mic, {
    silenceMs: 1200, onSpeechStart: () => events.push("start"),
    onSpeechEnd: (b) => events.push(`end:${b}`), ...opts,
  });
  return { mic, vad, events };
}

// 1. speech -> pause shorter than the silence window -> more speech -> long silence = ONE turn
{
  const { mic, vad, events } = make();
  feed(vad, mic, 0.005, 600);         // background noise only
  feed(vad, mic, 0.1, 900);           // speech
  feed(vad, mic, 0.005, 800);         // mid-sentence pause (< 1200 ms): turn continues
  feed(vad, mic, 0.1, 600);
  feed(vad, mic, 0.005, 1300);        // real end of turn
  await Promise.resolve();
  assert.deepEqual(events, ["start", "end:blob"]);
  assert.equal(mic.starts, 1, "recording starts once, on the first loud frame");
}

// 2. a short blip (< 150 ms) is not speech and its tentative recording is dropped
{
  const { mic, vad, events } = make();
  feed(vad, mic, 0.005, 300);
  feed(vad, mic, 0.2, 60);            // a cough / click
  feed(vad, mic, 0.005, 300);
  assert.deepEqual(events, []);
  assert.equal(mic.starts, 1);
  assert.equal(mic.stops, 1);
  assert.equal(mic.recording, false);
}

// 3. paused (while the reply is being prepared) -> loud input is ignored
{
  const { mic, vad, events } = make();
  vad.pause();
  feed(vad, mic, 0.2, 1000);
  assert.deepEqual(events, []);
  vad.resume();
  feed(vad, mic, 0.2, 300);
  assert.deepEqual(events, ["start"]);
}

// 4. while the coach speaks, the bar is higher: leaked coach audio doesn't trigger barge-in
{
  let coach = true;
  const { mic, vad, events } = make({ isCoachSpeaking: () => coach });
  feed(vad, mic, 0.005, 600);
  feed(vad, mic, 0.018, 600);         // above the normal threshold, below the doubled one
  assert.deepEqual(events, []);
  feed(vad, mic, 0.1, 300);           // the learner really talks over the coach
  assert.deepEqual(events, ["start"]);
  coach = false;
}

// 5. adaptive noise floor: in a noisy room, the same level no longer counts as speech
{
  const quiet = make();
  feed(quiet.vad, quiet.mic, 0.002, 3000);    // silent room
  feed(quiet.vad, quiet.mic, 0.04, 300);
  assert.deepEqual(quiet.events, ["start"], "0.04 is speech in a quiet room");

  const noisy = make();
  feed(noisy.vad, noisy.mic, 0.02, 3000);     // steady fan noise, below the starting threshold
  assert.ok(noisy.vad.threshold() > 0.05, "the threshold rose with the background");
  feed(noisy.vad, noisy.mic, 0.04, 300);
  assert.deepEqual(noisy.events, [], "0.04 is just noise in that room");
}

console.log("vad tests passed");
