"""The analyzers that run on every learner turn (in the background thread).

default_analyzers(cfg) decides which ones run. The two model-based upgrades
are switched in config.json, and each one quietly drops out if its model
isn't downloaded, so the session still works with the classical analyzers:
    "analyzers": {"contextual": true, "grammar": true}
"""

from ..errors import Unavailable
from .base import Analyzer, AnalyzerResult, Finding
from .fluency import FluencyAnalyzer
from .grammar import GrammarAnalyzer
from .vocabulary import VocabularyAnalyzer


def default_analyzers(cfg: dict | None = None) -> list[Analyzer]:
    """Register analyzers here. Adding a new one = one new line.
    `cfg` is the app config; optional analyzers read their on/off flags from it."""
    flags = (cfg or {}).get("analyzers", {})
    analyzers: list[Analyzer] = [
        # Contextual ranking loads DistilBERT lazily, inside the background
        # thread, and falls back to the baseline order if the model is missing.
        VocabularyAnalyzer(use_contextual=flags.get("contextual", False)),
        FluencyAnalyzer(),
    ]
    if flags.get("grammar", False):
        try:
            analyzers.append(GrammarAnalyzer())
        except Unavailable:
            pass  # the status screen already tells the learner how to enable it
    return analyzers


__all__ = ["Analyzer", "AnalyzerResult", "Finding", "FluencyAnalyzer", "GrammarAnalyzer",
           "VocabularyAnalyzer", "default_analyzers"]
