"""On-demand helpers the learner can press during a conversation.

  hint()       "What could I say?": three possible next replies at the
               learner's level. Both modes: it helps a stuck learner without
               anyone correcting them.
  say_better() "Say it better": a fluent version of the learner's sentence
               plus the main reason. Tutor mode only: it IS a correction, and
               roleplay characters never correct (the learner's rule).
  define()     the meaning of a word from the coach's reply, as used in that
               sentence. Both modes.

Each is ONE extra LLM call made only when the learner presses the button, so
it costs nothing otherwise. The answers are forced into a JSON schema
(OllamaChat.complete_json), so the UI always gets the fields it expects.
"""

from __future__ import annotations

from .llm import ChatModel
from .personas import LEVEL_GUIDE

HINT_SCHEMA = {
    "type": "object",
    "properties": {"suggestions": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3}},
    "required": ["suggestions"],
}
BETTER_SCHEMA = {
    "type": "object",
    "properties": {"better": {"type": "string"}, "why": {"type": "string"}},
    "required": ["better", "why"],
}
DEFINE_SCHEMA = {
    "type": "object",
    "properties": {"meaning": {"type": "string"}, "example": {"type": "string"},
                   "part_of_speech": {"type": "string"}},
    "required": ["meaning", "example", "part_of_speech"],
}


def _transcript(messages: list[dict], last: int = 8) -> str:
    """The recent conversation as plain text (system prompt left out)."""
    lines = []
    for m in [m for m in messages if m["role"] != "system"][-last:]:
        who = "Learner" if m["role"] == "user" else "Other person"
        lines.append(f"{who}: {m['content']}")
    return "\n".join(lines)


def hint(llm: ChatModel, messages: list[dict], level: str) -> list[str]:
    prompt = (
        "An English learner is in this conversation and doesn't know what to say next.\n\n"
        f"{_transcript(messages)}\n\n"
        f"Write 3 different things the LEARNER could say next, in the first person, each under 20 words. "
        f"Natural spoken English for CEFR level {level}: {LEVEL_GUIDE.get(level, '')} "
        "Make them different in content (for example: answer, ask back, add a detail). "
        "Do not write the other person's lines.")
    out = llm.complete_json([{"role": "user", "content": prompt}], HINT_SCHEMA, temperature=0.7)
    return [s.strip() for s in out.get("suggestions", []) if isinstance(s, str) and s.strip()][:3]


def say_better(llm: ChatModel, text: str, level: str) -> dict:
    prompt = (
        "A learner of English said the sentence below (from speech recognition, so ignore punctuation "
        "and capital letters).\n\n"
        f'Learner: "{text}"\n\n'
        f"Rewrite it the way a fluent speaker would say it, keeping the learner's meaning and a level "
        f"they can use (CEFR {level}). Then give the main improvement in under 20 words. "
        'If it is already natural, return it unchanged with why = "Already natural!".')
    out = llm.complete_json([{"role": "user", "content": prompt}], BETTER_SCHEMA, temperature=0.2)
    return {"better": str(out.get("better", "")).strip(), "why": str(out.get("why", "")).strip()}


def define(llm: ChatModel, word: str, sentence: str, level: str) -> dict:
    prompt = (
        f'Explain the word or phrase "{word}" as it is used in this sentence, for an English learner '
        f"at CEFR level {level}.\n\nSentence: \"{sentence}\"\n\n"
        "Give: the meaning in simple words (under 15 words), one new short example sentence, "
        "and the part of speech (noun, verb, adjective, adverb, phrase, ...).")
    out = llm.complete_json([{"role": "user", "content": prompt}], DEFINE_SCHEMA, temperature=0.2)
    return {k: str(out.get(k, "")).strip() for k in ("meaning", "example", "part_of_speech")}
