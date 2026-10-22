"""Characters are just config: one JSON file per persona in personas/.

Adding a character = adding a JSON file. The system prompt is built here so
every character follows the same coaching rules.

This module also decides WHICH characters a learner sees (by their goal) and
in WHICH accent each one speaks (by their accent setting).
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


@dataclass
class Persona:
    id: str
    name: str
    role: str
    goals: list[str]
    voice: str
    opening: str
    personality: str

    def system_prompt(self, level: str) -> str:
        level_line = LEVEL_GUIDE.get(level, LEVEL_GUIDE["B1"])
        return (f"You are {self.name}, {self.role}.\n{self.personality}\n\n"
                f"The learner's English level is {level}. {level_line}\n\n{RULES}")


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
    """Characters for this goal, plus the "general" ones everybody can use.
    No goal yet (before onboarding) -> everyone."""
    if not goal:
        return list(personas.values())
    return [p for p in personas.values() if goal in p.goals or "general" in p.goals]


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
