"""Pace analyzer: speaking speed and pauses, from Whisper's word timings.

Text alone can't show fluency in time: "I worked on a project" reads the same
whether it took 2 seconds or 9. For spoken turns, the voice pipeline puts each
word's start/end time on the Doc (doc._.timing, see nlp.py); typed turns have
none, and this analyzer then returns nothing.

Metrics (per spoken turn):
  words_per_minute   words / time from the first word's start to the last word's end.
                     Conversational English is roughly 120-160; learners often
                     speak slower, and very fast speech is hard to follow.
  pauses_per_minute  silent gaps of 250 ms or more between words. 250 ms is the
                     threshold that best predicted L2 fluency ratings in
                     De Jong & Bosker (2013), and a common choice in the field.
  mean_pause_s       average length of those pauses
  longest_pause_s
Findings: one per very long silence (1.5 s or more) mid-turn, placed on the word
after it, because that is where the learner got stuck.

Limits: Whisper's word timings are approximate (about +/- 0.1 s), and a pause
can be a learner thinking about content, not a language problem. So these are
coaching signals, not judgements.
"""

from __future__ import annotations

import re

from spacy.tokens import Doc

from .base import Analyzer, AnalyzerResult, Finding

PAUSE_S = 0.25        # shortest gap that counts as a pause
LONG_PAUSE_S = 1.5    # gaps this long get a finding
MIN_SPAN_S = 1.0      # don't compute a speed from less than a second of speech


class PaceAnalyzer(Analyzer):
    name = "pace"

    def analyze(self, doc: Doc) -> AnalyzerResult:
        words = [w for w in (doc._.timing or []) if w.get("word")]
        if len(words) < 2:
            return AnalyzerResult(self.name)             # typed turn, or a single word
        span = max(words[-1]["end"] - words[0]["start"], MIN_SPAN_S)
        gaps = [(round(b["start"] - a["end"], 2), b) for a, b in zip(words, words[1:])]
        pauses = [g for g, _ in gaps if g >= PAUSE_S]
        metrics = {
            "words_per_minute": round(len(words) / span * 60, 1),
            "pauses_per_minute": round(len(pauses) / span * 60, 1),
            "mean_pause_s": round(sum(pauses) / len(pauses), 2) if pauses else 0.0,
            "longest_pause_s": max(pauses, default=0.0),
        }
        findings = []
        offsets = _char_offsets(doc.text, words)
        for (gap, word), pos in zip(gaps, offsets[1:]):
            if gap >= LONG_PAUSE_S and pos is not None:
                start, end = pos
                findings.append(Finding(
                    self.name, "long_pause", doc.text[start:end], start, end,
                    f"You paused for {gap:.1f} s before “{doc.text[start:end]}”. If you need time to think, "
                    "a short bridge like “let me think…” keeps the conversation flowing.",
                    ["Let me think…", "That's a good question."]))
        return AnalyzerResult(self.name, findings, metrics)


def _char_offsets(text: str, words: list[dict]) -> list[tuple[int, int] | None]:
    """Where each timed word sits in the transcript text. Whisper's word list
    and its text normally agree, but if a word can't be found it gets None
    (no finding) rather than a wrong highlight."""
    out, cursor, lower = [], 0, text.lower()
    for w in words:
        token = re.sub(r"^[^\w']+|[^\w']+$", "", w["word"].lower())
        if not token:
            out.append(None)
            continue
        m = re.compile(rf"(?<![\w']){re.escape(token)}(?![\w'])").search(lower, cursor)
        if m:
            out.append((m.start(), m.end()))
            cursor = m.end()
        else:
            out.append(None)
    return out
