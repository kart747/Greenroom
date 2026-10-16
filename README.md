# English Coach (working name)

An offline, open-source English speaking coach. You talk to a character (a friendly
buddy, an HR interviewer, a visa officer); a local LLM plays the role while NLP
analyzers quietly measure your English in the background. At the end you get a
feedback report, and every session is saved so progress can be tracked over time.

Everything runs on your machine: no accounts, no cloud APIs.

## Status

| Layer | State |
|---|---|
| Text chat with characters (Ollama) | done |
| Vocabulary analyzer, classical baseline | done |
| Fluency analyzer (fillers, repetition, stats) | done |
| End-of-session report + saved sessions | done |
| Evaluation harness (precision / recall / F1) | done (starter test set) |
| Vocabulary analyzer, contextual BERT ranking | **TODO** (`contextual_suggestions`) |
| Hardware-based model config (`config.json`) | done |
| Push-to-talk voice (faster-whisper + Kokoro) | next |
| Live hands-free voice (VAD + barge-in) | later |
| Progress across sessions, goal onboarding | later |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
# install Ollama, then:
ollama pull qwen2.5:3b
```

## Configuration

Model choices live in `config.json` and are read in one place, `coach/config.py`.
Priority, lowest to highest: **defaults picked from your hardware** -> **config.json**
-> **command-line flags**. `config.json` only needs the keys you want to change.

```json
{
  "llm": {"model": "qwen2.5:3b", "host": "http://localhost:11434", "temperature": 0.7},
  "stt": {"whisper_size": "small", "device": "cuda", "compute_type": "float16"}
}
```

| Hardware | whisper size | device | compute type |
|---|---|---|---|
| NVIDIA GPU, >= 4 GB VRAM | small | cuda | float16 |
| NVIDIA GPU, 2-4 GB VRAM | base | cuda | float16 |
| CPU only | base | cpu | int8 |
| CPU only, < 6 GB RAM | tiny | cpu | int8 |

The GPU is shared with the LLM (qwen2.5:3b needs about 2.5 GB VRAM), which is why
whisper gets a smaller model than the card alone could hold.

```bash
python -m coach.config            # show detected hardware + effective config
python -m coach.config --write    # regenerate config.json for this machine
```

The LLM must stay at 5B parameters or less and run locally through Ollama.

## Run

```bash
python app.py                                   # choose a character
python app.py --persona hr_interviewer --level B1
python app.py --model llama3.2:3b               # any Ollama model (overrides config.json)
python app.py --config other_config.json
```

Type to talk. `/end` finishes and shows feedback; `/quit` exits.

```bash
python -m pytest -q          # tests (no Ollama needed, uses a fake LLM)
python -m eval.eval_vocab    # vocabulary detection precision/recall/F1
```

## How it works

```
learner turn ─┬─> coach LLM (in character) ──> streamed reply
              └─> spaCy preprocessing ──> analyzers (background thread)
                                             │
                     end of session  <───────┘  findings + metrics
                     -> report + sessions/*.json
```

- **Analyzers never sit in the reply path**, so they add zero latency.
- **The coach doesn't correct you mid-conversation** (it stays in character);
  feedback comes from the analyzers after the session.
- **Every analyzer follows one contract** (`coach/analyzers/base.py`): a spaCy
  `Doc` in, findings + metrics out. Adding grammar = one new file + one line in
  `coach/analyzers/__init__.py`.
- **Characters are JSON files** in `personas/`. Add a file to add a character.

## Project layout

```
app.py                     CLI entry point
config.json                model choices (whisper size, compute type, LLM)
coach/config.py            hardware detection + config loading (one place)
coach/nlp.py               shared spaCy preprocessing
coach/analyzers/           analyzer contract + vocabulary + fluency
coach/personas.py          persona loading + system prompt
coach/llm.py               streaming Ollama client
coach/session.py           conversation + background analysis
coach/report.py            aggregates findings into feedback
personas/*.json            characters
data/basic_words.json      overused-word lexicon (classical baseline)
eval/                      test set + evaluation script
tests/                     unit tests
```

## Known limitations of the baseline

- Synonyms are context-blind: the same suggestions appear in every sentence.
- Parser errors leak through, e.g. "left *for good* last year" gets flagged
  because spaCy reads "good" as describing "year".
- The evaluation set has 10 sentences; grow it to 50-100 before trusting numbers.
