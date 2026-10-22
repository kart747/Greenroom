"""Evaluate the vocabulary analyzer: DETECTION and SUGGESTION QUALITY.

Test set: eval/vocab_testset.jsonl, one learner sentence per line:
    {"text": "...", "weak": ["very good", ...],
     "acceptable": {"very good": ["great", "excellent"], ...}}
  weak        spans a human marked as weak in THIS sentence
  acceptable  for each weak span, the lexicon candidates that really fit THIS
              sentence (base forms; "got" -> ["receive"] also accepts "received")

Metrics
  detection    precision / recall / F1 of flagged spans vs "weak"
  accuracy@1   the FIRST suggestion is acceptable
  hit@3        at least one of the three shown suggestions is acceptable
               (both over correctly detected spans that have an acceptable list)
  ms/sentence  analysis time, to show what the BERT ranking costs

    python -m eval.eval_vocab                 # classical baseline, with error listing
    python -m eval.eval_vocab --contextual    # DistilBERT ranking
    python -m eval.eval_vocab --table         # both, as a markdown table for the README

Detection is the same for both versions (the BERT model only re-ranks
suggestions), so the interesting difference is in accuracy@1 and hit@3.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from coach.analyzers import VocabularyAnalyzer
from coach.nlp import preprocess

TESTSET = Path(__file__).parent / "vocab_testset.jsonl"


def load(path: Path = TESTSET) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def forms(base: str) -> set[str]:
    """A base word plus all its inflections: "receive" -> received, receives, ..."""
    out = {base.lower()}
    try:
        from lemminflect import getAllInflections
        for variants in getAllInflections(base).values():
            out.update(v.lower() for v in variants)
    except ImportError:
        pass
    return out


def evaluate(analyzer: VocabularyAnalyzer, examples: list[dict], verbose: bool = False) -> dict:
    tp = fp = fn = 0
    top1 = hit3 = rated = 0
    elapsed = 0.0
    for ex in examples:
        gold = {s.lower() for s in ex["weak"]}
        doc = preprocess(ex["text"])           # parsing isn't part of the analyzer's cost
        t = time.perf_counter()
        findings = analyzer.analyze(doc).findings
        elapsed += time.perf_counter() - t
        pred = {f.text.lower(): f for f in findings}
        tp += len(gold & pred.keys())
        fp += len(pred.keys() - gold)
        fn += len(gold - pred.keys())
        if verbose and (pred.keys() - gold or gold - pred.keys()):
            print(f"- {ex['text']}\n    false alarms: {sorted(pred.keys() - gold)}  missed: {sorted(gold - pred.keys())}")

        acceptable = {k.lower(): v for k, v in ex.get("acceptable", {}).items()}
        for span in gold & pred.keys():
            if not acceptable.get(span):
                continue
            ok = set().union(*(forms(a) for a in acceptable[span]))
            sugg = [s.lower() for s in pred[span].suggestions[:3]]
            rated += 1
            top1 += bool(sugg) and sugg[0] in ok
            hit3 += any(s in ok for s in sugg)
            if verbose and (not sugg or sugg[0] not in ok):
                print(f"  ~ '{span}' in: {ex['text']}\n      suggested {sugg}, acceptable {acceptable[span]}")

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"sentences": len(examples), "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
            "rated_spans": rated,
            "accuracy@1": round(top1 / rated, 3) if rated else 0.0,
            "hit@3": round(hit3 / rated, 3) if rated else 0.0,
            "ms_per_sentence": round(1000 * elapsed / len(examples), 1)}


def table(rows: dict[str, dict]) -> str:
    lines = ["| Version | Precision | Recall | F1 | Accuracy@1 | Hit@3 | ms/sentence |",
             "|---|---|---|---|---|---|---|"]
    for name, r in rows.items():
        lines.append(f"| {name} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
                     f"{r['accuracy@1']:.3f} | {r['hit@3']:.3f} | {r['ms_per_sentence']} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--contextual", action="store_true", help="rank suggestions with DistilBERT")
    ap.add_argument("--table", action="store_true", help="baseline vs contextual as a markdown table")
    args = ap.parse_args()
    examples = load()
    if args.table:
        base = evaluate(VocabularyAnalyzer(use_contextual=False), examples)
        bert = VocabularyAnalyzer(use_contextual=True)
        bert.analyze(preprocess("Warm up the model."))       # load BERT before timing
        ctx = evaluate(bert, examples)
        print(f"{len(examples)} sentences, {base['tp'] + base['fn']} weak spans, "
              f"{base['rated_spans']} with an acceptable-suggestion list\n")
        print(table({"Classical baseline": base, "DistilBERT ranking": ctx}))
        return
    print(json.dumps(evaluate(VocabularyAnalyzer(use_contextual=args.contextual), examples, verbose=True), indent=2))


if __name__ == "__main__":
    main()
