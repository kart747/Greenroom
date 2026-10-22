"""Evaluate the grammar analyzer against hand-marked learner errors.

Test set: eval/grammar_testset.jsonl
    {"text": "She don't like coffee.", "errors": [{"wrong": "don't", "fix": "doesn't"}]}
  "fix": "" means the words should be deleted. Some sentences have no errors
  at all, to catch the model "correcting" good English. A few are written like
  speech-recognition output (lowercase, no punctuation).

Metrics
  precision            flagged spans that overlap a real error
  recall               real errors that some flagged span overlaps
  F1
  correction accuracy  of the detected errors, how many got the RIGHT fix
                       (same words after the edit, ignoring case/punctuation)
  s/sentence           CPU time per sentence (runs in the background thread)

    python -m eval.eval_grammar            # list every miss and false alarm
    python -m eval.eval_grammar --table    # markdown table for the README
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from coach.analyzers.grammar import GrammarAnalyzer
from coach.nlp import preprocess

TESTSET = Path(__file__).parent / "grammar_testset.jsonl"


def load(path: Path = TESTSET) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def locate(text: str, wrong: str) -> tuple[int, int]:
    """Character span of the first whole-word match of `wrong` in `text`."""
    m = re.search(rf"(?<!\w){re.escape(wrong)}(?!\w)", text, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"'{wrong}' not found in: {text}")
    return m.start(), m.end()


def words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", s.lower().replace("’", "'"))


def apply(text: str, start: int, end: int, replacement: str) -> list[str]:
    return words(text[:start] + " " + replacement + " " + text[end:])


def evaluate(analyzer: GrammarAnalyzer, examples: list[dict], verbose: bool = False) -> dict:
    pred_total = pred_ok = gold_total = gold_found = fixed = 0
    elapsed = 0.0
    for ex in examples:
        text = ex["text"]
        doc = preprocess(text)
        t = time.perf_counter()
        findings = analyzer.analyze(doc).findings
        elapsed += time.perf_counter() - t
        gold = [(*locate(text, e["wrong"]), e["fix"]) for e in ex["errors"]]
        overlaps = lambda f, g: f.start < g[1] and g[0] < f.end   # noqa: E731

        pred_total += len(findings)
        pred_ok += sum(any(overlaps(f, g) for g in gold) for f in findings)
        for g in gold:
            gold_total += 1
            hits = [f for f in findings if overlaps(f, g)]
            if not hits:
                if verbose:
                    print(f"- MISSED '{text[g[0]:g[1]]}' -> '{g[2]}' in: {text}")
                continue
            gold_found += 1
            # The fix is right if applying the model's edit gives the same words
            # as applying the gold edit (span boundaries may differ slightly).
            f = hits[0]
            if apply(text, f.start, f.end, (f.suggestions or [""])[0]) == apply(text, g[0], g[1], g[2]):
                fixed += 1
            elif verbose:
                print(f"- WRONG FIX '{f.text}' -> {f.suggestions} (gold '{text[g[0]:g[1]]}' -> '{g[2]}') in: {text}")
        if verbose:
            for f in findings:
                if not any(overlaps(f, g) for g in gold):
                    print(f"- FALSE ALARM '{f.text}' -> {f.suggestions} in: {text}")

    precision = pred_ok / pred_total if pred_total else 0.0
    recall = gold_found / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"sentences": len(examples), "gold_errors": gold_total, "flagged": pred_total,
            "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
            "correction_accuracy": round(fixed / gold_found, 3) if gold_found else 0.0,
            "s_per_sentence": round(elapsed / len(examples), 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--table", action="store_true", help="markdown table for the README")
    args = ap.parse_args()
    analyzer = GrammarAnalyzer()
    analyzer.analyze(preprocess("Warm up the model."))     # load before timing
    r = evaluate(analyzer, load(), verbose=not args.table)
    if args.table:
        print(f"{r['sentences']} sentences, {r['gold_errors']} marked errors\n")
        print("| Model | Precision | Recall | F1 | Correction accuracy | s/sentence (CPU) |")
        print("|---|---|---|---|---|---|")
        print(f"| T5-base GEC + difflib | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
              f"{r['correction_accuracy']:.3f} | {r['s_per_sentence']} |")
    else:
        print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
