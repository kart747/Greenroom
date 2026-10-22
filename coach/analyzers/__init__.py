from .base import Analyzer, AnalyzerResult, Finding
from .fluency import FluencyAnalyzer
from .vocabulary import VocabularyAnalyzer


def default_analyzers(cfg: dict | None = None) -> list[Analyzer]:
    """Register analyzers here. Adding grammar later = one new line.
    `cfg` is the app config; optional analyzers will read their on/off flags from it."""
    return [VocabularyAnalyzer(), FluencyAnalyzer()]


__all__ = ["Analyzer", "AnalyzerResult", "Finding", "FluencyAnalyzer",
           "VocabularyAnalyzer", "default_analyzers"]
