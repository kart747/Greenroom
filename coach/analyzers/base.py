"""The analyzer contract.

Every analyzer takes a preprocessed spaCy Doc and returns an AnalyzerResult.
The session and report never care which analyzer produced a finding, so adding
a new analyzer (grammar, pronunciation, ...) means writing one new module and
registering it -- nothing else changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field

from spacy.tokens import Doc


@dataclass
class Finding:
    analyzer: str          # which analyzer produced this, e.g. "vocabulary"
    kind: str              # finer category, e.g. "basic_word", "filler"
    text: str              # the exact span from the learner's speech
    start: int             # character offsets into the turn text
    end: int
    message: str           # human-readable explanation
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnalyzerResult:
    analyzer: str
    findings: list[Finding] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "analyzer": self.analyzer,
            "findings": [f.to_dict() for f in self.findings],
            "metrics": self.metrics,
        }


class Analyzer(ABC):
    name: str = "base"

    @abstractmethod
    def analyze(self, doc: Doc) -> AnalyzerResult:
        """Analyze one learner turn. Must be fast and side-effect free."""
