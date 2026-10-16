"""Text chat with an English coach character. Voice comes in the next layer.

Usage:
    python app.py                       # pick a character interactively
    python app.py --persona hr_interviewer --level B1 --model qwen2.5:3b
    python app.py --config my_config.json

Model choices come from config.json (see coach/config.py); flags override it.

During the chat:  /end -> finish and show feedback,  /quit -> exit without feedback
"""

from __future__ import annotations

import argparse
from pathlib import Path

from coach.config import CONFIG_PATH, load_config
from coach.llm import OllamaChat
from coach.personas import load_personas
from coach.report import render_text, summarize
from coach.session import Session

SESSIONS_DIR = Path(__file__).parent / "sessions"


def pick_persona(personas: dict):
    print("Choose who you want to talk to:")
    items = list(personas.values())
    for i, p in enumerate(items, 1):
        print(f"  {i}. {p.name} - {p.role}")
    while True:
        choice = input("> ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(items):
            return items[int(choice) - 1]
        print(f"Enter a number from 1 to {len(items)}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline English speaking coach (text mode)")
    parser.add_argument("--persona", help="persona id, e.g. buddy, hr_interviewer, visa_officer")
    parser.add_argument("--level", default="B1", choices=["A2", "B1", "B2", "C1"])
    # default=None so an unset flag doesn't override config.json (load_config skips Nones)
    parser.add_argument("--model", default=None, help="Ollama model name (overrides config.json)")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="path to config.json")
    args = parser.parse_args()
    cfg = load_config(args.config, overrides={"llm": {"model": args.model}})

    personas = load_personas()
    persona = personas.get(args.persona) if args.persona else pick_persona(personas)
    if persona is None:
        raise SystemExit(f"Unknown persona '{args.persona}'. Options: {', '.join(personas)}")

    llm_cfg = cfg["llm"]
    llm = OllamaChat(model=llm_cfg["model"], host=llm_cfg["host"], temperature=llm_cfg["temperature"])
    session = Session(persona, llm, level=args.level)
    print(f"\n{persona.name}: {session.opening}\n")

    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            text = "/end"
        if not text:
            continue
        if text == "/quit":
            return
        if text == "/end":
            break
        print(f"{persona.name}: ", end="", flush=True)
        try:
            for chunk in session.user_turn(text):
                print(chunk, end="", flush=True)
        except RuntimeError as err:
            print(f"\n[error] {err}")
            return
        print("\n")

    record = session.finish()
    if not record["turns"]:
        print("No turns recorded.")
        return
    print(render_text(summarize(record)))
    path = Session.save(record, SESSIONS_DIR)
    print(f"\nSession saved to {path}")


if __name__ == "__main__":
    main()
