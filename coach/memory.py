"""What the tutor remembers about the learner, across conversations.

Two kinds of memory, both turned into a short paragraph that goes into the
tutor's system prompt (never into roleplay characters, see personas.py):

  1. facts the learner told the tutor ("I'm a final-year CS student",
     "my IELTS exam is in March"). After each tutor chat, the LLM picks out
     at most 5 such facts from the LEARNER's own messages; they are stored in
     SQLite (newest 20 kept). The learner can read and delete them in Settings.
  2. recurring mistakes, counted from the analyzers' findings over all saved
     sessions ("very good" x5, "am having" -> "have"). No LLM involved: these
     numbers come straight from the reports.

So the tutor can say "Last time you mentioned your interview at TCS. How did
it go?" or bring up a word the learner keeps overusing.
"""

from __future__ import annotations

import re
from collections import Counter

from .llm import ChatModel
from .storage import Storage

FACTS_SCHEMA = {
    "type": "object",
    "properties": {"facts": {"type": "array", "items": {"type": "string"}, "maxItems": 5}},
    "required": ["facts"],
}

EXTRACT_PROMPT = """You help an English tutor remember a student between lessons.
Below are things the STUDENT said in today's conversation.
List at most 5 short facts the student shared about themselves that a tutor should remember:
studies, job, goals, exams and their dates, interests, upcoming events, where they live.
Rules:
- Only facts the student clearly stated about THEMSELVES. A sentence they asked about or wanted
  corrected is not a fact about them.
- One fact per item, under 15 words, written in the third person, using the student's own words.
- Skip anything about English mistakes, and facts already known.
Return {"facts": []} if there is nothing new.

Already known:
%s

Student's messages:
%s"""


def recurring_mistakes(storage: Storage, min_count: int = 2) -> tuple[list[tuple[str, int]], list[str]]:
    """(overused words with counts, grammar fixes like '"am having" -> "have"')."""
    words: Counter = Counter()
    grammar: Counter = Counter()
    fixes: dict[str, str] = {}
    for s in storage.list_sessions():
        words.update(s["summary"].get("counts", {}).get("vocabulary", {}))
        for issue in s["summary"].get("top_issues", {}).get("grammar", []):
            grammar[issue["text"]] += issue["count"]
            if issue["suggestions"]:
                fixes[issue["text"]] = issue["suggestions"][0]
    overused = [(w, n) for w, n in words.most_common(5) if n >= min_count]
    # Only grammar slips with a known fix and more than one word of context:
    # "was" -> "were" alone means nothing to the tutor out of context.
    slips = [f'"{t}" -> "{fixes[t]}"' for t, _ in grammar.most_common(10)
             if t in fixes and fixes[t] and len(t.split()) + len(fixes[t].split()) > 2][:3]
    return overused, slips


def learner_context(profile: dict | None, storage: Storage) -> str:
    """The paragraph the tutor's system prompt gets about this learner."""
    profile = profile or {}
    lines = ["About the learner (use this naturally; never read it out as a list):"]
    goal = (profile.get("goal") or "general").replace("_", " ")
    who = f"- Name: {profile.get('name', 'unknown')}. Practising for: {goal}"
    if profile.get("goal") == "study_abroad" and profile.get("destination"):
        who += f" (destination: {profile['destination'].title()})"
    lines.append(who + ".")
    facts = [m["text"] for m in storage.list_memories()]
    if facts:
        lines.append("- Things they told you before: " + "; ".join(facts) + ".")
    overused, slips = recurring_mistakes(storage)
    if overused:
        lines.append("- Words they overuse: " + ", ".join(f'"{w}" ({n}x)' for w, n in overused)
                     + ". Encourage stronger alternatives when it fits.")
    if slips:
        lines.append("- Grammar slips seen in earlier sessions: " + "; ".join(slips) + ".")
    return "\n".join(lines)


_STOP = {"student", "they", "their", "them", "with", "from", "that", "this", "about", "will", "would",
         "have", "has", "had", "been", "into", "also", "very", "currently", "wants", "likes"}


def grounded(fact: str, source: str) -> bool:
    """Keep a fact only if most of its content words appear in what the
    learner actually said. Small LLMs sometimes invent or copy facts (one
    copied an example out of the prompt); this checks the answer against the
    source text instead of trusting it. Words are compared by their first 5
    letters, so "studies" matches "studying"."""
    stem = lambda w: w[:5]                                       # noqa: E731
    said = {stem(w) for w in re.findall(r"[a-z0-9]+", source.lower())}
    words = [w for w in re.findall(r"[a-z0-9]+", fact.lower()) if len(w) > 3 and w not in _STOP]
    if not words:
        return False
    return sum(stem(w) in said for w in words) / len(words) >= 0.5


def remember(llm: ChatModel, storage: Storage, record: dict) -> list[str]:
    """After a tutor chat: extract new facts from the learner's messages and
    store them. Runs in a background thread when the chat closes, so the
    learner never waits for it. Returns the new facts (for tests/logs)."""
    said = [t["text"] for t in record["turns"]]
    if not said:
        return []
    known = [m["text"] for m in storage.list_memories()]
    prompt = EXTRACT_PROMPT % ("\n".join(f"- {k}" for k in known) or "(nothing yet)",
                               "\n".join(f"- {s}" for s in said[-30:]))
    try:
        facts = llm.complete_json([{"role": "user", "content": prompt}], FACTS_SCHEMA,
                                  temperature=0.1)["facts"]
    except (RuntimeError, KeyError, TypeError):
        return []                      # memory is a nice-to-have; never break closing a chat
    known_lower = {k.lower() for k in known}
    source = " ".join(said)
    new = [f.strip() for f in facts if isinstance(f, str) and f.strip()
           and f.strip().lower() not in known_lower and grounded(f, source)][:5]
    if new:
        storage.add_memories(new)
    return new
