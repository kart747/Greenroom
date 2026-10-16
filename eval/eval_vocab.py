"""Evaluate weak-word DETECTION of the vocabulary analyzer.

Each test line lists the spans a human marked as weak. We compare what the
analyzer flags against that gold list and report precision / recall / F1.

    python -m eval.eval_vocab                 # classical baseline
    python -m eval.eval_vocab --contextual    # your BERT version, once implemented

Grow the test set to 50-100 sentences (ideally real learner transcripts) before
trusting the numbers. Suggestion QUALITY needs a second eval (human ratings or an
"acceptable replacements" list per span) -- add that when the BERT ranker exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from coach.analyzers import VocabularyAnalyzer
from coach.nlp import preprocess

TESTSET = Path(__file__).parent / "vocab_testset.jsonl"


def evaluate(analyzer: VocabularyAnalyzer, path: Path = TESTSET, verbose: bool = False) -> dict:
    tp = fp = fn = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        ex = json.loads(line)
        gold = {s.lower() for s in ex["weak"]}
        pred = {f.text.lower() for f in analyzer.analyze(preprocess(ex["text"])).findings}
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
        if verbose and (pred - gold or gold - pred):
            print(f"- {ex['text']}\n    false alarms: {sorted(pred - gold)}  missed: {sorted(gold - pred)}")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 3),
            "recall": round(recall, 3), "f1": round(f1, 3)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--contextual", action="store_true")
    args = ap.parse_args()
    print(evaluate(VocabularyAnalyzer(use_contextual=args.contextual), verbose=True))
