"""The local web server behind the Greenroom app (FastAPI).

It only listens on 127.0.0.1, so nothing outside this computer can reach it.
The UI in web/ is plain HTML/CSS/JS served from the same server, so there is
no build step and no CDN.

API overview (all JSON unless noted):
  GET  /api/status                  which engines work (LLM, STT, TTS)
  GET  /api/profile, PUT /api/profile
  GET  /api/home                    greeting data: due cards, last session stats
  GET  /api/personas                characters for the learner's goal, with resolved voices
  POST /api/sessions                start: opening line + its audio
  POST /api/sessions/{id}/turn      one turn (typed text or uploaded audio) -> NDJSON stream
  POST /api/sessions/{id}/abort     barge-in: stop the coach's current reply
  POST /api/sessions/{id}/end       finish: report, saved to the database
  GET  /api/reports/{db_id}         reopen an old report
  GET  /api/progress                trends over all sessions
  GET  /api/review/due              flashcards due now
  POST /api/review/{card_id}        grade a card (0/3/4/5), reschedules it with SM-2

Why the turn endpoint streams NDJSON (one JSON object per line): the browser
can act on each event as soon as it arrives (show the transcript, play the
first sentence) instead of waiting for the whole reply. Audio travels as
base64 WAV inside those lines, which keeps it one simple stream.

Why each turn runs in its own worker thread: if the learner interrupts
(barge-in) or closes the window, the browser drops the connection. The worker
still finishes or stops cleanly and releases the session, so the next turn
never finds the conversation half-updated.
"""

from __future__ import annotations

import base64
import json
import queue
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .analyzers import default_analyzers
from . import srs
from .engines import Engines
from .errors import Unavailable
from .exercises import build_exercises
from .personas import GOALS, Persona, load_personas, personas_for_goal, resolve_voice
from .progress import progress
from .report import report_payload, summarize
from .session import Session
from .storage import Storage
from .voice import VoicePipeline

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
LEVELS = ["A2", "B1", "B2", "C1"]
ACCENTS = ["american", "british", "auto"]


@dataclass
class LiveSession:
    """A conversation that is still going on (kept in memory until it ends)."""
    session: Session
    pipeline: VoicePipeline
    # One turn at a time per conversation; the next turn waits for this.
    lock: threading.Lock = field(default_factory=threading.Lock)
    # Set by /abort; the running turn checks it between events and stops.
    abort: threading.Event = field(default_factory=threading.Event)


class ProfileIn(BaseModel):
    name: str
    goal: str
    level: str
    accent: str = "auto"
    destination: str | None = None
    silence_ms: int = 1200      # hands-free: how long a pause ends the learner's turn


class StartIn(BaseModel):
    persona_id: str


class GradeIn(BaseModel):
    grade: int


def create_app(cfg: dict, engines: Engines, storage: Storage,
               personas: dict[str, Persona] | None = None) -> FastAPI:
    app = FastAPI(title="Greenroom", docs_url=None, redoc_url=None)
    personas = personas or load_personas()
    live: dict[str, LiveSession] = {}

    # ---- status + profile ---------------------------------------------
    @app.get("/api/status")
    def status():
        return engines.status()

    @app.get("/api/profile")
    def get_profile():
        return {"profile": storage.get_profile(), "goals": GOALS, "levels": LEVELS, "accents": ACCENTS}

    @app.put("/api/profile")
    def set_profile(p: ProfileIn):
        if p.goal not in GOALS or p.level not in LEVELS or p.accent not in ACCENTS:
            raise HTTPException(422, "Unknown goal, level or accent.")
        if not p.name.strip():
            raise HTTPException(422, "Please enter your name.")
        if not 400 <= p.silence_ms <= 5000:
            raise HTTPException(422, "Silence threshold must be between 400 and 5000 ms.")
        data = p.model_dump()
        data["name"] = data["name"].strip()
        storage.set_profile(data)
        return {"profile": data}

    @app.get("/api/home")
    def home():
        sessions = storage.list_sessions()
        last = _named(sessions[-1]) if sessions else None
        return {"profile": storage.get_profile(), "due_cards": storage.count_due(),
                "session_count": len(sessions), "last_session": last}

    # ---- characters ---------------------------------------------------
    @app.get("/api/personas")
    def list_personas():
        profile = storage.get_profile()
        goal = (profile or {}).get("goal")
        return [_persona_json(p, profile) for p in personas_for_goal(personas, goal)]

    # ---- conversation -------------------------------------------------
    @app.post("/api/sessions")
    def start_session(body: StartIn):
        persona = personas.get(body.persona_id)
        if persona is None:
            raise HTTPException(404, "Unknown character.")
        try:
            engines.llm.check()
        except Unavailable as err:
            raise HTTPException(503, str(err)) from err
        profile = storage.get_profile() or {}
        session = Session(persona, engines.llm, level=profile.get("level", "B1"),
                          analyzers=default_analyzers(cfg))
        pipeline = VoicePipeline(session, engines.stt, engines.tts, voice=resolve_voice(persona, profile))
        sid = uuid.uuid4().hex
        live[sid] = LiveSession(session, pipeline)
        opening = _encode(pipeline.opening())
        return {"id": sid, "persona": _persona_json(persona, profile), "opening": opening}

    @app.post("/api/sessions/{sid}/turn")
    def turn(sid: str, text: str | None = Form(None), audio: UploadFile | None = File(None)):
        ls = _live(sid)
        data = audio.file.read() if audio is not None else None
        if data is None and not (text and text.strip()):
            raise HTTPException(422, "Send text or audio.")
        events: queue.Queue = queue.Queue()
        threading.Thread(target=_run_turn, args=(ls, data, text, events), daemon=True).start()

        def stream():
            while (ev := events.get()) is not None:
                yield json.dumps(_encode(ev)) + "\n"
        return StreamingResponse(stream(), media_type="application/x-ndjson")

    @app.post("/api/sessions/{sid}/abort")
    def abort(sid: str):
        _live(sid).abort.set()
        return {"ok": True}

    @app.post("/api/sessions/{sid}/end")
    def end_session(sid: str):
        ls = _live(sid)
        ls.abort.set()                  # ending mid-reply: stop the coach first
        with ls.lock:                   # ...and wait for that turn to wrap up
            record = ls.session.finish()
        live.pop(sid, None)
        if not record["turns"]:
            return {"empty": True}
        summary = summarize(record)
        summary["exercises"] = build_exercises(record)
        db_id = storage.save_session(record, summary)
        _make_cards(storage, record)
        return _named({"id": db_id, **report_payload(record, summary)})

    @app.get("/api/reports/{db_id}")
    def get_report(db_id: int):
        stored = storage.get_session(db_id)
        if stored is None:
            raise HTTPException(404, "No such session.")
        return _named({"id": db_id, **report_payload(stored["record"], stored["summary"])})

    # ---- progress + review --------------------------------------------
    @app.get("/api/progress")
    def get_progress():
        p = progress(storage.list_sessions())
        p["sessions"] = [_named(s) for s in p["sessions"]]
        return p

    @app.get("/api/review/due")
    def review_due():
        return {"cards": storage.due_cards()}

    @app.post("/api/review/{card_id}")
    def grade_card(card_id: int, body: GradeIn):
        card = storage.get_card(card_id)
        if card is None:
            raise HTTPException(404, "No such card.")
        if body.grade not in srs.GRADES:
            raise HTTPException(422, "Grade must be 0, 3, 4 or 5.")
        ease, interval, reps, due = srs.review(card["ease"], card["interval"], card["reps"], body.grade)
        storage.update_schedule(card_id, ease, interval, reps, due)
        return {"id": card_id, "ease": ease, "interval": interval, "reps": reps, "due": due}

    # ---- helpers ------------------------------------------------------
    def _named(item: dict) -> dict:
        """Add the character's display name ("Mr. Rao") next to its id."""
        p = personas.get(item["persona"])
        return {**item, "persona_name": p.name if p else item["persona"].replace("_", " ")}

    def _live(sid: str) -> LiveSession:
        ls = live.get(sid)
        if ls is None:
            raise HTTPException(404, "This conversation has ended. Start a new one.")
        return ls

    # The UI. Mounted last so /api/* routes win.
    if WEB_DIR.is_dir():
        @app.get("/")
        def index():
            return FileResponse(WEB_DIR / "index.html")
        app.mount("/", StaticFiles(directory=WEB_DIR), name="web")
    return app


def _run_turn(ls: LiveSession, audio: bytes | None, text: str | None, events: queue.Queue) -> None:
    """Worker thread: run one turn and push its events; None marks the end."""
    with ls.lock:
        ls.abort.clear()   # an abort meant for the previous turn must not kill this one
        gen = ls.pipeline.run_turn(audio=audio, text=text)
        try:
            for ev in gen:
                if ls.abort.is_set():
                    events.put({"type": "aborted"})
                    break
                events.put(ev)
        except Unavailable as err:
            events.put({"type": "error", "message": str(err)})
        except ValueError as err:          # undecodable recording
            events.put({"type": "error", "message": str(err)})
        except RuntimeError as err:        # Ollama went away mid-reply
            events.put({"type": "error", "message": str(err)})
        finally:
            gen.close()    # on abort: stops the LLM request and keeps the partial reply
            events.put(None)


def _make_cards(storage: Storage, record: dict) -> None:
    """Every flagged vocabulary span becomes a flashcard (or refreshes its
    example if the card already exists). The learner's own sentence is the
    example, so the card reminds them where they used the weak word."""
    for turn in record["turns"]:
        for res in turn["results"]:
            if res["analyzer"] != "vocabulary":
                continue
            for f in res["findings"]:
                if f["suggestions"]:
                    storage.add_card(f["text"], {"suggestions": f["suggestions"], "message": f["message"],
                                                 "example": turn["text"], "kind": f["kind"]})


def _encode(ev: dict) -> dict:
    """Audio bytes -> base64 text so the event can travel as JSON."""
    if ev.get("wav") is not None:
        ev = {**ev, "wav": base64.b64encode(ev["wav"]).decode("ascii")}
    return ev


def _persona_json(p: Persona, profile: dict | None) -> dict:
    voice = resolve_voice(p, profile)
    return {"id": p.id, "name": p.name, "role": p.role, "goals": p.goals,
            "voice": voice, "accent": voice.split("_")[0], "opening": p.opening}
