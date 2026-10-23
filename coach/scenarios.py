"""Custom scenarios: the learner describes a situation, the LLM designs the
other person in it, and that becomes a roleplay character like the built-in ones.

    "I need to ask my manager for two days of leave"   -> a manager to persuade
    a pasted job description (kind="job")               -> an interviewer for that job

Only the CHARACTER comes from the LLM (name, role, how they behave, their first
line). The rules every roleplay follows (stay in character, never correct the
learner, short spoken replies) are still added by Persona.system_prompt(), so
a generated character can't switch them off.

The answer is forced into a JSON schema and every field is checked and
trimmed before use, because a small model's output is never trusted blindly.
"""

from __future__ import annotations

from .llm import ChatModel
from .personas import Persona

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "role": {"type": "string"},
        "personality": {"type": "string"},
        "opening": {"type": "string"},
        "gender": {"type": "string", "enum": ["female", "male"]},
    },
    "required": ["name", "role", "personality", "opening", "gender"],
}

SITUATION_PROMPT = """An English learner wants to rehearse a real conversation. Their situation:
\"\"\"{text}\"\"\"

Design the OTHER person in this conversation (never the learner), for a voice roleplay:
- name: a realistic name with a title if natural (e.g. a surname with Mr./Ms./Dr.)
- role: who they are, under 10 words
- personality: 2-4 sentences of instructions for an actor playing them: how they
  behave, what they want, what they will ask or push back on. Realistic, not rude.
- opening: their first line to the learner, 1-2 short sentences, inviting a reply
- gender: female or male"""

JOB_PROMPT = """An English learner is preparing for a job interview. The job description:
\"\"\"{text}\"\"\"

Design the INTERVIEWER for this job, for a voice roleplay:
- name: a realistic name with a title (Mr./Ms.)
- role: e.g. "hiring manager for the <job title> role", under 10 words
- personality: 2-4 sentences of instructions: professional and polite; asks one
  question at a time about the skills, tools and duties in this job description,
  the learner's experience and motivation; follows up when an answer is vague.
- opening: a greeting and the first question, 1-2 sentences
- gender: female or male"""

KINDS = {"situation": SITUATION_PROMPT, "job": JOB_PROMPT}
MAX_INPUT = 3000        # characters of description / job ad sent to the model


def design(llm: ChatModel, text: str, kind: str = "situation") -> dict:
    """Ask the LLM for a character and return checked, trimmed fields."""
    prompt = KINDS[kind].format(text=text.strip()[:MAX_INPUT])
    out = llm.complete_json([{"role": "user", "content": prompt}], SCHEMA, temperature=0.6, max_tokens=400)
    clean = {k: " ".join(str(out.get(k, "")).split()) for k in ("name", "role", "personality", "opening")}
    clean["name"] = clean["name"][:40] or "Your partner"
    clean["role"] = clean["role"][:80] or "a conversation partner"
    clean["personality"] = clean["personality"][:900]
    clean["opening"] = clean["opening"][:300] or "Hello! Shall we begin?"
    clean["gender"] = out.get("gender") if out.get("gender") in ("female", "male") else "female"
    clean["kind"] = kind
    clean["description"] = text.strip()[:MAX_INPUT]
    return clean


def to_persona(scenario_id: int, data: dict) -> Persona:
    """A stored scenario as a roleplay Persona (American voice of the right
    gender; the learner's accent setting can still change the accent)."""
    return Persona(id=f"custom_{scenario_id}", name=data["name"], role=data["role"], goals=["custom"],
                   voice=f"american_{data['gender']}", opening=data["opening"],
                   personality=data["personality"], mode="roleplay")
