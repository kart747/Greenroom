"""Three practice tasks after each session, built from the session's top issues.

Every task comes from a fixed template per finding kind, filled with the
learner's own words, the analyzer's suggestions and a topic that suits the
word. No LLM writes them: a template is predictable, never invents a wrong
"correction", and is easy to check. That matters more here than variety.

    "very good" (intensified_basic) ->
    "Describe your favourite food in two sentences without using 'very' or 'good'."
"""

from __future__ import annotations

from collections import Counter

# A topic where the word naturally comes up, so the learner has to use the
# stronger alternative instead.
WORD_TOPICS = {
    "good": "your favourite food", "nice": "your hometown", "bad": "a day when everything went wrong",
    "big": "your city", "small": "your city", "happy": "a moment you'll never forget",
    "sad": "a moment you'll never forget", "interesting": "a film or series you watched recently",
    "boring": "a film or series you watched recently", "beautiful": "the most beautiful place you've visited",
    "hard": "your toughest exam", "easy": "your toughest exam", "thing": "what you carry in your bag every day",
    "stuff": "what you carry in your bag every day", "get": "how you got your first phone",
    "important": "what matters most to you in a job",
}
DEFAULT_TOPIC = "how you spent last weekend"
QUESTIONS = ["What are your plans for the next five years?", "Why did you choose your course?",
             "What do you do to relax after a long day?"]


def collect_issues(record: dict) -> list[dict]:
    """All findings of a session, grouped by (kind, span), most frequent first."""
    counts: Counter = Counter()
    first: dict[tuple[str, str], dict] = {}
    for turn in record["turns"]:
        for res in turn["results"]:
            for f in res["findings"]:
                key = (f["kind"], f["text"].lower())
                counts[key] += 1
                # keep the first sentence it appeared in, as the example
                first.setdefault(key, {**f, "sentence": turn["text"]})
    return [{**first[k], "count": n} for k, n in counts.most_common()]


def build_exercises(record: dict, n: int = 3) -> list[dict]:
    """Pick up to n issues, one per kind first (variety), then the most frequent."""
    issues = collect_issues(record)
    picked, kinds = [], set()
    for issue in issues:                      # pass 1: different kinds of problem
        if issue["kind"] not in kinds:
            picked.append(issue)
            kinds.add(issue["kind"])
    for issue in issues:                      # pass 2: fill up with the rest
        if issue not in picked:
            picked.append(issue)
    return [make_exercise(issue, i) for i, issue in enumerate(picked[:n])]


def make_exercise(issue: dict, i: int = 0) -> dict:
    kind, text = issue["kind"], issue["text"]
    sugg = [s for s in issue.get("suggestions", []) if s]
    words = text.lower().split()
    base = words[-1]                          # the basic word itself ("good" in "very good")
    topic = WORD_TOPICS.get(_lemma_guess(base), DEFAULT_TOPIC)
    alt = " or ".join(f"“{s}”" for s in sugg[:2])

    if kind == "intensified_basic":
        title = "One strong word"
        prompt = f"Describe {topic} in two sentences without using “{words[0]}” or “{base}”."
        tip = f"Try {alt} instead." if alt else ""
    elif kind == "basic_word":
        title = "Swap a basic word"
        prompt = f"Say two sentences about {topic} using {alt} instead of “{text}”." if alt else \
                 f"Say two sentences about {topic} without using “{text}”."
        tip = f"You said: “{issue['sentence']}”"
    elif kind == "filler":
        title = "Pause instead of “" + text + "”"
        prompt = f"Talk for 30 seconds about {DEFAULT_TOPIC}. Each time you want to say “{text}”, pause silently instead."
        tip = "A short silence sounds calm and confident; listeners don't notice it."
    elif kind == "filler_phrase":
        title = f"Drop “{text}”"
        prompt = f"Answer in three sentences, without saying “{text}”: {QUESTIONS[i % len(QUESTIONS)]}"
        tip = "Record yourself in the next session and listen for it in the report."
    elif kind == "long_pause":
        title = "Keep talking through a pause"
        prompt = (f"Answer in about 45 seconds, without going silent for more than a second: "
                  f"{QUESTIONS[i % len(QUESTIONS)]}")
        tip = "When you need time, say “let me think…” or “that's a good question” instead of stopping."
    elif kind == "repetition":
        title = "Say it smoothly"
        smooth = _remove_repetition(issue["sentence"], text)
        prompt = f"Read this aloud slowly, then at normal speed, without restarting: “{smooth}”"
        tip = "If you lose a word, keep going instead of repeating the last one."
    elif kind == "grammar":
        fix = sugg[0] if sugg else ""
        corrected = _apply(issue["sentence"], issue["start"], issue["end"], fix)
        title = "Fix the sentence"
        prompt = f"Say the corrected sentence three times: “{corrected}”"
        tip = (f"Then make a new sentence of your own with “{fix}”." if fix
               else f"Notice that “{text}” is not needed.")
    else:
        title, prompt, tip = "Practise", f"Make two new sentences without “{text}”.", ""
    return {"kind": kind, "target": text, "title": title, "prompt": prompt, "tip": tip}


def _lemma_guess(word: str) -> str:
    """Cheap lemma for topic lookup only ("got" -> "get", "things" -> "thing")."""
    irregular = {"got": "get", "gets": "get", "getting": "get", "gotten": "get",
                 "bigger": "big", "biggest": "big", "smaller": "small", "smallest": "small",
                 "better": "good", "best": "good", "worse": "bad", "worst": "bad",
                 "happier": "happy", "easier": "easy", "harder": "hard"}
    if word in irregular:
        return irregular[word]
    return word[:-1] if word.endswith("s") and word[:-1] in WORD_TOPICS else word


def _remove_repetition(sentence: str, repeated: str) -> str:
    """'I I worked hard' + 'I I' -> 'I worked hard'."""
    first = repeated.split()[0]
    return sentence.replace(repeated, first, 1)


def _apply(sentence: str, start: int, end: int, fix: str) -> str:
    """Put the correction into the sentence. Offsets are relative to the turn,
    and the stored sentence IS the turn text, so they line up."""
    out = sentence[:start] + fix + sentence[end:]
    return " ".join(out.split())
