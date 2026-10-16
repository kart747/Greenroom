from .base import Analyzer, AnalyzerResult, Finding
from .fluency import FluencyAnalyzer
from .vocabulary import VocabularyAnalyzer


def default_analyzers() -> list[Analyzer]:
    """Register analyzers here. Adding grammar later = one new line."""
    return [VocabularyAnalyzer(), FluencyAnalyzer()]


__all__ = ["Analyzer", "AnalyzerResult", "Finding", "FluencyAnalyzer",
           "VocabularyAnalyzer", "default_analyzers"]
