"""Pronunciation feedback: which words and sounds came out differently.

How it works (no cloud, all CPU):
  1. EXPECTED sounds: espeak-ng turns the target text into IPA phones, word by
     word ("think" -> θ ɪ ŋ k). Two references, American and British English,
     so a learner isn't marked down for following either one.
  2. HEARD sounds: a wav2vec2 model fine-tuned to output phones instead of
     words (facebook/wav2vec2-xlsr-53-espeak-cv-ft, Apache-2.0). It writes
     what was actually said: "tɪŋk" if the learner said "tink". Whisper can't
     do this, because it's trained to guess the intended WORD and quietly fixes
     mispronunciations.
  3. ALIGN the two phone sequences with edit distance (like diff, for sounds).
     Each expected phone is then: right, close (a near sound), or wrong/missing.
  4. SCORE each word (share of its sounds that were right) and the sentence, and
     explain the most common confusions with a tip ("th" in think: tongue
     between the teeth).

Accent fairness, on purpose: features of Indian (and other) English that
don't change meaning are NOT errors: a tapped or rolled r, aspirated or dental
t, a dropped "r" after a vowel (as in British English). Contrasts that change
meaning are: thin/tin, very/wery, ship/sheep, full/fool, cat/ket.

Limits: the phone recognizer itself makes mistakes, especially on noisy audio,
and the scores are evaluated on speechocean762 (eval/eval_pronunciation.py),
whose speakers are Mandarin speakers of English, not Indian. So this is
coaching, not a test score.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import numpy as np

from .errors import Unavailable

HF_DIR = Path(__file__).resolve().parents[1] / "models" / "hf"
PHONE_MODEL = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
SAMPLE_RATE = 16_000
WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")

# ---- comparing phones ----------------------------------------------------------
# r-coloured vowels are split into vowel + r, so British (no r) and American
# (with r) references compare cleanly.
# The vowel in "bird"/"work" is written ɜː, ɝ or ɚ depending on the accent and
# the tool; it becomes "ɜ + optional r", and ɜ counts as the same sound as ə
# (see cost()), so all spellings compare equal.
SPLIT = {"ɑːɹ": ["ɑː", "ɹ"], "ɔːɹ": ["ɔː", "ɹ"], "oːɹ": ["oː", "ɹ"], "ɛɹ": ["ɛ", "ɹ"], "ɪɹ": ["ɪ", "ɹ"],
         "ʊɹ": ["ʊ", "ɹ"], "ɚ": ["ə", "ɹ"], "ɝ": ["ɜ", "ɹ"], "ɜː": ["ɜ", "ɹ"],
         "aɪɚ": ["aɪ", "ə", "ɹ"], "aɪə": ["aɪ", "ə"], "əl": ["ə", "l"], "iə": ["ɪ", "ə"]}
# Marks and variants that don't change meaning in English.
STRIP = str.maketrans("", "", "ʰ̪ʲ̃ˈˌ")
SAME = {"ɐ": "ə", "ᵻ": "ɪ", "ɫ": "l", "ɾ": "t", "r": "ɹ", "ʁ": "ɹ", "ɻ": "ɹ", "ɽ": "ɹ", "ʔ": "t",
        "e": "eɪ", "o": "oʊ", "oː": "oʊ", "əʊ": "oʊ", "ɒ": "ɑː", "ɑ": "ɑː", "ɔ": "ɔː",
        "ʈ": "t", "ɖ": "d", "ɳ": "n", "ɭ": "l", "i": "iː", "u": "uː", "a": "æ", "x": "k"}
# Close pairs: understandable but not quite (half a point). The common
# Indian-English "v/w as one sound" (ʋ) lands here.
CLOSE = {frozenset(p) for p in [("v", "ʋ"), ("w", "ʋ"), ("d", "t"), ("ð", "d"), ("z", "s"), ("ʒ", "ʃ"),
                                  ("ə", "ʌ"), ("ə", "ɪ"), ("ə", "ɛ"), ("ɛ", "eɪ"), ("ɔː", "oʊ"), ("ɑː", "ʌ"),
                                  ("ɪ", "ɛ"), ("l", "ʊ"), ("l", "o")]}   # last two: a "dark l" at the end of "people"
VOWELS = set("aeiouæɑɒɔəɛɜɪʊʌɐᵻ")

# What each sound is, for learners ("th as in think" beats "/θ/").
NAMES = {"θ": "“th” as in think", "ð": "“th” as in this", "ʃ": "“sh” as in she", "ʒ": "“s” as in measure",
         "tʃ": "“ch” as in chair", "dʒ": "“j” as in job", "ŋ": "“ng” as in sing", "ɹ": "“r” as in red",
         "j": "“y” as in yes", "w": "“w” as in we", "v": "“v” as in very", "z": "“z” as in zoo",
         "æ": "“a” as in cat", "ɛ": "“e” as in bed", "ɪ": "short “i” as in ship", "iː": "long “ee” as in sheep",
         "ʊ": "short “oo” as in full", "uː": "long “oo” as in fool", "ʌ": "“u” as in cup",
         "ɑː": "“a” as in father", "ɔː": "“aw” as in law", "ə": "weak “uh” as in about",
         "ɜ": "“ir” as in bird", "eɪ": "“ay” as in day", "aɪ": "“i” as in my", "ɔɪ": "“oy” as in boy",
         "aʊ": "“ow” as in now", "oʊ": "“o” as in go", "h": "“h” as in hat"}
TIPS = {
    "θ": "Put the tip of your tongue between your teeth and blow gently: “th” in think, not “t”.",
    "ð": "Tongue between your teeth, with your voice on: “th” in this, not “d”.",
    "v": "Top teeth lightly on your lower lip: “v” in very. Your lips shouldn't round.",
    "w": "Round your lips like a small “oo” with no teeth touching: “w” in we.",
    "z": "Let it buzz like a bee: “z” in is and zoo, not a hiss.",
    "ʃ": "Round your lips and pull your tongue back: “sh” in she, not “s”.",
    "iː": "Make it longer and smile a little: sheep, not ship.",
    "ɪ": "Keep it short and relaxed: ship, not sheep.",
    "uː": "Longer, with rounded lips: fool, not full.",
    "ʊ": "Short and relaxed: full, not fool.",
    "æ": "Open your mouth wider: cat, not ket.",
    "ʌ": "A short, relaxed “uh”: cup.",
    "ŋ": "Back of the tongue up, no “g” after it: sing, not sin-g.",
}


def canon(phones: list[str]) -> list[str]:
    """Normalise phones before comparing (split r-coloured vowels, drop marks)."""
    out = []
    for p in phones:
        for q in SPLIT.get(p, [p]):
            q = q.translate(STRIP)
            out.append(SAME.get(q, q))
    return [p for p in out if p]


EQUAL = {frozenset(("ɜ", "ə"))}   # "bird" vowel vs r-coloured schwa: the same sound


def cost(expected: str, heard: str) -> float:
    if expected == heard or frozenset((expected, heard)) in EQUAL:
        return 0.0
    if frozenset((expected, heard)) in CLOSE:
        return 0.5
    return 1.0


def align(expected: list[str], heard: list[str], optional: list[bool]) -> list[tuple[str | None, float]]:
    """Edit-distance alignment. For each expected phone: (heard phone or None
    if missing, score 1 / 0.5 / 0). An optional phone (a post-vowel "r") may be
    missing for free."""
    n, m = len(expected), len(heard)
    INS = 0.7     # an extra heard sound costs a little less than a wrong one
    d = np.zeros((n + 1, m + 1))
    d[1:, 0] = np.cumsum([0.0 if o else 1.0 for o in optional])
    d[0, 1:] = np.arange(1, m + 1) * INS
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j - 1] + cost(expected[i - 1], heard[j - 1]),
                          d[i - 1, j] + (0.0 if optional[i - 1] else 1.0),
                          d[i, j - 1] + INS)
    out: list[tuple[str | None, float]] = [(None, 0.0)] * n
    i, j = n, m
    while i > 0:
        if j > 0 and d[i, j] == d[i - 1, j - 1] + cost(expected[i - 1], heard[j - 1]):
            out[i - 1] = (heard[j - 1], 1.0 - cost(expected[i - 1], heard[j - 1]))
            i, j = i - 1, j - 1
        elif d[i, j] == d[i - 1, j] + (0.0 if optional[i - 1] else 1.0):
            out[i - 1] = (None, 1.0 if optional[i - 1] else 0.0)
            i -= 1
        else:
            j -= 1
    return out


# ---- expected phones (espeak) -------------------------------------------------
_g2p_lock = threading.Lock()
_backends: dict[str, object] = {}


def _backend(lang: str):
    with _g2p_lock:
        if lang not in _backends:
            try:
                import espeakng_loader
                from phonemizer.backend import EspeakBackend
                from phonemizer.backend.espeak.wrapper import EspeakWrapper
            except ImportError as exc:
                raise Unavailable("Pronunciation off: espeak/phonemizer is missing (installed with kokoro-onnx).") from exc
            EspeakWrapper.set_library(espeakng_loader.get_library_path())
            EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
            _backends[lang] = EspeakBackend(lang, preserve_punctuation=False, with_stress=False,
                                            language_switch="remove-flags")
        return _backends[lang]


def expected_phones(words: list[str], lang: str = "en-us") -> list[list[str]]:
    """IPA phones for each word, split into the recognizer's phone inventory."""
    from phonemizer.separator import Separator
    sep = Separator(phone=" ", word="", syllable="")
    out = _backend(lang).phonemize(words, separator=sep, strip=True)
    return [p.split() for p in out]


# ---- heard phones (wav2vec2) -------------------------------------------------
class PhoneRecognizer:
    """wav2vec2 phone recognition on the CPU, loaded from models/hf/ only."""

    def __init__(self, model_name: str = PHONE_MODEL, cache_dir: Path = HF_DIR):
        try:
            import torch
            from huggingface_hub import hf_hub_download
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
        except ImportError as exc:
            raise Unavailable("Pronunciation off: transformers/torch are not installed.") from exc
        try:
            local = {"cache_dir": str(cache_dir), "local_files_only": True}
            self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, **local)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name, **local).eval()
            vocab = json.loads(Path(hf_hub_download(model_name, "vocab.json", **local)).read_text())
        except Exception as exc:
            raise Unavailable("Pronunciation off: the phone model is missing. Run: python setup_models.py") from exc
        self.id_to_phone = {i: p for p, i in vocab.items()}
        self.torch = torch

    def recognize(self, audio: np.ndarray) -> list[str]:
        """16 kHz float samples -> phones, by greedy CTC decoding: take the most
        likely symbol per 20 ms frame, merge repeats, drop the blank symbol."""
        inputs = self.extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with self.torch.no_grad():
            ids = self.model(inputs.input_values).logits[0].argmax(-1).tolist()
        phones, prev = [], None
        for i in ids:
            if i != prev:
                p = self.id_to_phone.get(i, "")
                if p and not p.startswith("<"):
                    phones.append(p)
            prev = i
        return phones


_lock = threading.Lock()
_recognizer: PhoneRecognizer | None = None
_error: Unavailable | None = None


def get_recognizer() -> PhoneRecognizer:
    """Load the phone model once per process (~1.3 GB in memory); remember failures."""
    global _recognizer, _error
    with _lock:
        if _error is not None:
            raise _error
        if _recognizer is None:
            try:
                _recognizer = PhoneRecognizer()
            except Unavailable as err:
                _error = err
                raise
        return _recognizer


# ---- scoring -----------------------------------------------------------------
def score_words(text: str, heard: list[str], g2p=expected_phones) -> dict:
    """Compare what was heard with the expected sounds of `text` (American and
    British references; each word takes the kinder one).
    Returns {"score": 0-100, "words": [...]} with per-word details."""
    spans = [(m.group(), m.start(), m.end()) for m in WORD.finditer(text)]
    words = [w for w, _, _ in spans]
    refs = [[canon(p) for p in g2p(words, lang)] for lang in ("en-us", "en-gb")] if words else []
    return score_against(spans, refs, heard)


def score_against(spans: list[tuple[str, int, int]], refs: list[list[list[str]]], heard: list[str],
                  allow_optional: bool = True) -> dict:
    """The scoring core: `refs` holds one or more references, each a list of
    phones per word. The evaluation passes the dataset's own pronunciations
    here, with allow_optional=False so every phone keeps its human label."""
    heard_c = canon(heard)
    per_ref = []
    for exp in refs:
        flat = [p for w in exp for p in w]
        owner = [k for k, w in enumerate(exp) for _ in w]
        # Optional sounds, missing for free:
        #  - an "r" after a vowel and not before one (British and Indian English drop it)
        #  - the "uh" before a word-final l/n/m: "people" is often "peepl" (a syllabic l)
        last = [i + 1 == len(flat) or owner[i + 1] != owner[i] for i in range(len(flat))]
        optional = [allow_optional and (
                        (p == "ɹ" and i > 0 and flat[i - 1][0] in VOWELS and
                         (i + 1 == len(flat) or flat[i + 1][0] not in VOWELS))
                        or (p == "ə" and i + 1 < len(flat) and last[i + 1] and flat[i + 1] in ("l", "n", "m")))
                    for i, p in enumerate(flat)]
        per_ref.append((flat, owner, optional, align(flat, heard_c, optional)))

    result_words = []
    total, count = 0.0, 0
    for k, (word, start, end) in enumerate(spans):
        best = None
        for flat, owner, optional, aligned in per_ref:
            idx = [i for i, o in enumerate(owner) if o == k and not optional[i]]
            if not idx:
                continue
            s = sum(aligned[i][1] for i in idx) / len(idx)
            if best is None or s > best[0]:      # the kinder of the references wins
                best = (s, [{"expected": flat[i], "heard": aligned[i][0], "score": aligned[i][1]} for i in idx])
        if best is None:
            continue
        s, phones = best
        total += s * len(phones)
        count += len(phones)
        result_words.append({"word": word, "start": start, "end": end, "score": round(s, 2),
                             "status": "good" if s >= 0.8 else "close" if s >= 0.5 else "work",
                             "phones": phones, "tip": _tip(phones)})
    return {"score": round(100 * total / count) if count else 0, "words": result_words}


def _tip(phones: list[dict]) -> str:
    """One helpful sentence about this word's weakest sound."""
    wrong = [p for p in phones if p["score"] < 1.0]
    if not wrong:
        return ""
    worst = min(wrong, key=lambda p: p["score"])
    exp = worst["expected"]
    name = NAMES.get(exp, f"/{exp}/")
    if worst["heard"] is None:
        msg = f"The {name} sound was missing."   # NAMES never start with "the"
    else:
        msg = f"I heard {NAMES.get(worst['heard'], '/' + worst['heard'] + '/')} instead of {name}."
    return f"{msg} {TIPS.get(exp, '')}".strip()


def assess(audio: np.ndarray, text: str, recognizer: PhoneRecognizer | None = None,
           g2p=None) -> dict:
    """Score a recording of someone reading `text` aloud. Tests pass a fake
    recognizer and dictionary (g2p)."""
    rec = recognizer or get_recognizer()
    heard = rec.recognize(audio)
    if len(heard) < 2:
        return {"score": 0, "words": [], "heard_phones": heard, "empty": True}
    return {**score_words(text, heard, g2p or expected_phones), "heard_phones": heard, "empty": False}
