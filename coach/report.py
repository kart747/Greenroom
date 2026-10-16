"""Turns a session record into a learner-facing feedback report.

Works only on the analyzers' JSON output, so it doesn't care which analyzers ran.
"""

from __future__ import annotations

from collections import Counter, defaultdict


def summarize(record: dict) -> dict:
    """Aggregate findings and metrics across all turns of a session."""
    repeated = defaultdict(Counter)      # analyzer -> Counter of normalized spans
    suggestions: dict[str, list[str]] = {}
    messages: dict[str, str] = {}
    metric_sums = defaultdict(lambda: defaultdict(float))
    metric_counts = defaultdict(int)

    for turn in record["turns"]:
        for res in turn["results"]:
            name = res["analyzer"]
            metric_counts[name] += 1
            for k, v in res["metrics"].items():
                metric_sums[name][k] += v
            for f in res["findings"]:
                key = f["text"].lower()
                repeated[name][key] += 1
                suggestions.setdefault(key, f["suggestions"])
                messages.setdefault(key, f["message"])

    averages = {name: {k: round(v / metric_counts[name], 2) for k, v in sums.items()}
                for name, sums in metric_sums.items()}
    top = {name: [{"text": text, "count": n, "message": messages[text],
                   "suggestions": suggestions[text]}
                  for text, n in counts.most_common(5)]
           for name, counts in repeated.items()}
    return {"turns": len(record["turns"]), "averages": averages, "top_issues": top}


def render_text(summary: dict) -> str:
    lines = ["", "=" * 50, "  SESSION FEEDBACK", "=" * 50,
             f"You spoke {summary['turns']} time(s)."]
    for name, issues in summary["top_issues"].items():
        lines.append(f"\n[{name.upper()}]")
        for i in issues:
            times = f" (x{i['count']})" if i["count"] > 1 else ""
            lines.append(f"  - \"{i['text']}\"{times}: {i['message']}")
            if i["suggestions"]:
                lines.append(f"      try: {', '.join(i['suggestions'])}")
    if not summary["top_issues"] or not any(summary["top_issues"].values()):
        lines.append("\nNo issues found. Nice work!")
    lines.append("\n[STATS]")
    for name, metrics in summary["averages"].items():
        stats = ", ".join(f"{k}={v}" for k, v in metrics.items())
        lines.append(f"  {name}: {stats}")
    return "\n".join(lines)
