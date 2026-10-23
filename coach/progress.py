"""Progress across sessions: the numbers behind the Progress screen's charts.

Each stored session already has per-analyzer averages (report.summarize), so
progress is just those averages lined up in time order, plus a tally of the
words that got flagged most often across ALL sessions.
"""

from __future__ import annotations

from collections import Counter


def progress(sessions: list[dict]) -> dict:
    """sessions: Storage.list_sessions() output, oldest first."""
    series = []
    flagged: Counter = Counter()
    for s in sessions:
        avg = s["summary"].get("averages", {})
        vocab, fluency = avg.get("vocabulary", {}), avg.get("fluency", {})
        series.append({
            "id": s["id"],
            "started": s["started"],
            "persona": s["persona"],
            "turns": s["summary"].get("turns", 0),
            # averages are per turn, so basic_word_count here = basic words per turn
            "basic_words_per_turn": vocab.get("basic_word_count"),
            "type_token_ratio": vocab.get("type_token_ratio"),
            "fillers_per_100_words": fluency.get("fillers_per_100_words"),
            "grammar_errors_per_turn": avg.get("grammar", {}).get("error_count"),
            "words_per_minute": avg.get("pace", {}).get("words_per_minute"),
            "pauses_per_minute": avg.get("pace", {}).get("pauses_per_minute"),
            "latency_total_s": s.get("latency", {}).get("total_s"),
        })
        # Only vocabulary spans: they are the words worth drilling (and they
        # become flashcards). Fillers ("um") and grammar spans ("was", "a")
        # would crowd the list without being words to learn.
        flagged.update(s["summary"].get("counts", {}).get("vocabulary", {}))
    return {
        "sessions": series,
        "most_flagged": [{"text": t, "count": n} for t, n in flagged.most_common(10)],
    }
