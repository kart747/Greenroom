"""Characters are just config: one JSON file per persona in personas/.

Adding a character = adding a JSON file. The system prompt is built here so
every character follows the same coaching rules.

This module also decides WHICH characters a learner sees (by their goal) and
in WHICH accent each one speaks (by their accent setting).

Two kinds of persona ("mode" in the JSON):
  roleplay  a character in a real situation (HR interviewer, visa officer...).
            Stays in character and never corrects the learner; feedback comes
            after the session.
  tutor     Sam, the English tutor (personas/tutor.json). Answers questions
            about English, chats, and corrects mistakes during the chat. This
            is the ONLY place in the app where in-chat correction happens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PERSONA_DIR = Path(__file__).resolve().parents[1] / "personas"

LEVEL_GUIDE = {
    "A2": "Use very simple, common words and short sentences.",
    "B1": "Use everyday vocabulary and clear, medium-length sentences.",
    "B2": "Use natural, varied vocabulary at a normal pace.",
    "C1": "Speak naturally, including idioms and nuanced vocabulary.",
}

RULES = """Rules:
- Stay in character the whole time. Never mention that you are an AI or a language tutor.
- Keep every reply to 1-3 short spoken sentences. This is a voice conversation.
- Respond to what the learner said, then keep the conversation going (usually with a question).
- Do NOT correct the learner's English. Detailed feedback is given separately after the session.
  If they make a mistake, you may naturally repeat the correct form in your reply, without pointing it out.
- No lists, markdown, emojis or stage directions -- only words you would say out loud."""

# How much the tutor corrects; the learner picks this in Settings.
TUTOR_CORRECTIONS = {
    "gentle": ("If the learner's message has an important grammar or word-choice mistake, "
               "correct only the MOST important one, briefly, then reply to what they said."),
    "thorough": ("Correct every clear grammar or word-choice mistake in the learner's message "
                 "(at most three), briefly, then reply to what they said."),
    "off": "Do not correct the learner's mistakes unless they ask you to.",
}

TUTOR_RULES = """How to help:
- If the learner asks about English (grammar, a word, how to say something), answer clearly
  with one or two short example sentences, simple enough for their level.
- If they just chat, chat back naturally and keep the conversation going with a question.
- Corrections: {corrections}
  Format a correction like this: You said "I am having two years experience" -> **I have two years of experience**.
  Ignore punctuation and capital letters: the learner's words come from speech recognition.
- Keep replies short: 2-4 sentences when chatting, at most about 120 words when explaining.
  Your replies are also read aloud.
- You may use **bold** and short "- " lists for examples. No headings, tables or code.
- Never use emojis or emoticons.
- Be warm and encouraging; mistakes are normal. You are an AI tutor in the Greenroom app."""


@dataclass
class Persona:
    id: str
    name: str
    role: str
    goals: list[str]
    voice: str
    opening: str
    personality: str
    mode: str = "roleplay"      # "roleplay" or "tutor", see the module docstring

    def system_prompt(self, level: str, learner: str = "", corrections: str = "gentle") -> str:
        """`learner` (tutor only) is what the tutor remembers about this learner,
        from coach/memory.py. Roleplay characters don't get it: a real
        interviewer doesn't know your hobbies, and that realism is the point."""
        level_line = LEVEL_GUIDE.get(level, LEVEL_GUIDE["B1"])
        head = (f"You are {self.name}, {self.role}.\n{self.personality}\n\n"
                f"The learner's English level is {level}. {level_line}\n\n")
        if self.mode != "tutor":
            return head + RULES
        rules = TUTOR_RULES.format(corrections=TUTOR_CORRECTIONS.get(corrections, TUTOR_CORRECTIONS["gentle"]))
        return head + rules + (f"\n\n{learner}" if learner else "")

    def opening_for(self, profile: dict | None) -> str:
        """The first line, with the learner's name filled in where the JSON has {name}."""
        name = (profile or {}).get("name") or "there"
        return self.opening.replace("{name}", name)


def load_personas(directory: Path = PERSONA_DIR) -> dict[str, Persona]:
    personas = {}
    for path in sorted(directory.glob("*.json")):
        p = Persona(**json.loads(path.read_text()))
        personas[p.id] = p
    return personas


GOALS = ["placements", "study_abroad", "job", "general"]

# Where a study-abroad learner is going -> the accent they will hear there.
DESTINATION_ACCENT = {
    "uk": "british", "ireland": "british", "australia": "british", "new zealand": "british",
    "usa": "american", "us": "american", "canada": "american",
}


def personas_for_goal(personas: dict[str, Persona], goal: str | None) -> list[Persona]:
    """Roleplay characters for this goal, plus the "general" ones everybody can
    use. No goal yet (before onboarding) -> every character. The tutor is not a
    roleplay character, so it is never in this list."""
    roleplay = [p for p in personas.values() if p.mode == "roleplay"]
    if not goal:
        return roleplay
    return [p for p in roleplay if goal in p.goals or "general" in p.goals]


def resolve_voice(persona: Persona, profile: dict | None) -> str:
    """Pick the voice key (e.g. "british_female") for this learner.

    1. An explicit accent choice (american / british) wins.
    2. "auto" + study_abroad -> the accent of the destination country,
       because that is the English the learner must get used to.
    3. Otherwise the persona's own default.
    The persona's gender is always kept: Mr. Rao stays male whatever the accent.
    """
    profile = profile or {}
    default_accent, gender = persona.voice.split("_", 1)
    accent = profile.get("accent", "auto")
    if accent not in ("american", "british"):
        accent = default_accent
        if profile.get("goal") == "study_abroad":
            dest = (profile.get("destination") or "").strip().lower()
            accent = DESTINATION_ACCENT.get(dest, default_accent)
    return f"{accent}_{gender}"
