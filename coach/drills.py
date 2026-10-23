"""Speaking drills on review cards: say the correct form, and get it checked.

Seeing a correction isn't the same as producing it. Each review card asks the
learner to SAY (or type) the better version, which language-teaching research
calls "pushed output". The answer goes through speech recognition and is
checked here with simple, explainable word rules (no LLM):

  grammar card     original "I am having two years experience", fix "am having" -> "have"
                   correct = the fix's words appear in the answer AND the
                   mistake doesn't, checked WITH its neighbouring words
                   ("there is many", not just "is", which can be right elsewhere).
                   A deletion fix, "discuss about" -> "discuss", only needs the
                   extra word gone.
  vocabulary card  weak "very good", suggestions ["excellent", "great", ...]
                   correct = one of the suggestions appears, in any form
                   (lemminflect: "receive" also accepts "received").

Words are compared lowercase without punctuation, because the answer comes
from speech recognition.
"""

from __future__ import annotations

import re

WORD = re.compile(r"[a-z0-9']+")


def words(text: str) -> list[str]:
    return WORD.findall(text.lower().replace("’", "'"))


def contains(seq: list[str], part: list[str]) -> bool:
    """Is `part` a contiguous run of words inside `seq`?"""
    n = len(part)
    return n > 0 and any(seq[i:i + n] == part for i in range(len(seq) - n + 1))


def apply_fix(sentence: str, start: int, end: int, fix: str) -> str:
    """The learner's sentence with the correction put in."""
    return " ".join((sentence[:start] + fix + sentence[end:]).split())


def grammar_card(finding: dict, sentence: str) -> tuple[str, dict]:
    """(front key, card data) for a grammar finding. The key names the error
    pattern ("am having -> have"), so the same mistake in another session
    refreshes one card instead of creating duplicates."""
    fix = (finding.get("suggestions") or [""])[0]
    wrong = finding["text"]
    key = f"{wrong.lower()} -> {fix.lower() or '(leave out)'}"
    before, after = words(sentence[:finding["start"]])[-1:], words(sentence[finding["end"]:])[:1]
    return key, {"kind": "grammar", "wrong": wrong, "fix": fix, "start": finding["start"],
                 "end": finding["end"], "example": sentence, "message": finding["message"],
                 "corrected": apply_fix(sentence, finding["start"], finding["end"], fix),
                 "context": [before, after], "suggestions": [fix] if fix else []}


def check(card: dict, answer: str) -> dict:
    """Is the learner's answer right for this card? Returns
    {"correct": bool, "expected": what a right answer contains, "why": short reason}."""
    said = words(answer)
    if card.get("kind") == "grammar":
        wrong, fix = words(card["wrong"]), words(card["fix"])
        before, after = card.get("context") or ([], [])
        mistake_gone = not wrong or contains(fix, wrong) or not contains(said, before + wrong + after)
        if fix:
            ok = contains(said, fix) and mistake_gone
            why = (f'Right: “{card["fix"]}”, not “{card["wrong"]}”.' if ok
                   else f'Use “{card["fix"]}” instead of “{card["wrong"]}”.')
        else:
            ok = mistake_gone and len(said) >= 2
            why = f'Right: no “{card["wrong"]}” needed.' if ok else f'Leave out “{card["wrong"]}”.'
        return {"correct": ok, "expected": card["corrected"], "why": why}

    # vocabulary card: any suggestion, in any form
    forms = set()
    for s in card.get("suggestions", []):
        forms |= _forms(s)
    hit = next((f for f in forms if contains(said, words(f))), None)
    weak = words(card["front"])
    return {"correct": hit is not None,
            "expected": " / ".join(card.get("suggestions", [])),
            "why": (f"Nice: “{hit}”." if hit else f"Try a stronger word than “{' '.join(weak)}”.")}


def _forms(word: str) -> set[str]:
    out = {word.lower()}
    try:
        from lemminflect import getAllInflections, getAllLemmas
        base = word.lower()
        for lemmas in getAllLemmas(base).values():
            base = lemmas[0]
            break
        out.add(base)
        for variants in getAllInflections(base).values():
            out.update(v.lower() for v in variants)
    except ImportError:
        pass
    return out
