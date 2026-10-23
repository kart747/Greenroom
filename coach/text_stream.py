"""Cut a stream of LLM tokens into whole sentences.

Why: text-to-speech needs a complete sentence to sound natural, but waiting for
the WHOLE reply would add seconds of silence. So we speak sentence 1 while the
LLM is still writing sentence 2. The coach starts talking as soon as its first
sentence is finished.

    chunker = SentenceChunker()
    for token in llm_tokens:
        for sentence in chunker.feed(token):
            speak(sentence)
    if rest := chunker.flush():
        speak(rest)
"""

from __future__ import annotations

import re

# Words that end with "." but do not end a sentence. Compared lowercase,
# without the final dot, e.g. "Mr." -> "mr", "e.g." -> "e.g".
ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "mt",
    "e.g", "i.e", "approx", "dept", "fig",
}

# Sentence end = one or more of . ! ? (optionally followed by a closing quote or
# bracket), and then whitespace. Requiring the whitespace matters: a token
# stream can stop at "3." and only the next token tells us whether it was
# "3.5" (a decimal) or the end of a sentence. A newline also ends a sentence.
_END = re.compile(r"[.!?]+[\"'”’)\]]*(?=\s)|\n+")


class SentenceChunker:
    def __init__(self) -> None:
        self._buf = ""

    def feed(self, token: str) -> list[str]:
        """Add a token; return the sentences that are now complete (maybe none)."""
        self._buf += token
        sentences = []
        start = 0
        for m in _END.finditer(self._buf):
            if m.group().startswith(".") and self._is_abbreviation(self._buf[start:m.start()]):
                continue
            sentence = self._buf[start:m.end()].strip()
            if sentence:
                sentences.append(sentence)
            start = m.end()
        self._buf = self._buf[start:]
        return sentences

    def flush(self) -> str | None:
        """Return whatever is left at the end of the reply (the last sentence
        usually has no whitespace after its full stop)."""
        rest, self._buf = self._buf.strip(), ""
        return rest or None

    @staticmethod
    def _is_abbreviation(text_before_dot: str) -> bool:
        words = text_before_dot.split()
        if not words:
            return False
        word = words[-1].lstrip("\"'“‘([").lower()
        # A single capital letter is an initial ("J. K. Rowling"), except "I".
        if len(word) == 1 and word.isalpha() and word != "i":
            return True
        return word in ABBREVIATIONS


_MARKUP = re.compile(r"\*\*|__|`|^\s*[-*•]\s+|^\s*#+\s*", re.MULTILINE)


def speakable(text: str) -> str:
    """Remove the light markdown the tutor may use (**bold**, "- " list
    bullets) so the voice doesn't read out symbols. The tutor writes a
    correction as  You said "x" -> **y**,  which should sound like
    'You said "x". Better: y'."""
    return _MARKUP.sub("", text).replace(" -> ", ". Better: ").replace("->", ". Better: ").strip()
