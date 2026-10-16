"""Characters are just config: one JSON file per persona in personas/.

Adding a character = adding a JSON file. The system prompt is built here so
every character follows the same coaching rules.
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
