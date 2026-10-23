"""Local storage: one SQLite file at userdata/greenroom.db (gitignored).

Why SQLite: it ships with Python, needs no server, and one file per user keeps
the "everything stays on your machine" promise. Progress charts need queries
over many sessions, which is awkward with a folder of JSON files.

Tables
  profile       one row: the onboarding answers (name, goal, level, accent, ...)
  sessions      one row per conversation (roleplay or tutor chat): the full
                record + its summary as JSON, a title for the sidebar
  review_cards  flashcards for flagged words, with their spaced-repetition schedule
  memory        short facts the tutor remembers about the learner ("works in Pune")
  scenarios     roleplay characters the learner created from a description

Each finished session is ALSO exported to sessions/*.json (the same file the
CLI used to write), so a session can be opened or shared without the database.

A new connection is opened per call: FastAPI runs request handlers on a thread
pool, and a sqlite3 connection must not be shared between threads.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "userdata" / "greenroom.db"
EXPORT_DIR = ROOT / "sessions"

SCHEMA = """
CREATE TABLE IF NOT EXISTS profile (
    id      INTEGER PRIMARY KEY CHECK (id = 1),   -- single learner per install
    data    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    started  REAL NOT NULL,
    persona  TEXT NOT NULL,
    level    TEXT NOT NULL,
    record   TEXT NOT NULL,      -- full session record (turns, findings, transcript, latency)
    summary  TEXT NOT NULL       -- report.summarize() output + exercises
);
CREATE TABLE IF NOT EXISTS scenarios (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    data     TEXT NOT NULL,                 -- name, role, personality, opening, gender, kind, description
    hidden   INTEGER NOT NULL DEFAULT 0,    -- "deleted": kept so old reports still show the name
    created  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS memory (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    text     TEXT NOT NULL UNIQUE,
    created  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS review_cards (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    front     TEXT NOT NULL UNIQUE,  -- the flagged span, lowercase ("very good")
    data      TEXT NOT NULL,         -- suggestions, message, example sentence
    ease      REAL NOT NULL DEFAULT 2.5,
    interval  INTEGER NOT NULL DEFAULT 0,   -- days until the next review
    reps      INTEGER NOT NULL DEFAULT 0,   -- successful reviews in a row
    due       REAL NOT NULL,                -- unix time
    created   REAL NOT NULL
);
"""


class Storage:
    def __init__(self, path: Path = DB_PATH, export_dir: Path | None = EXPORT_DIR):
        self.path = path
        self.export_dir = export_dir
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.executescript(SCHEMA)
            self._migrate(db)

    @staticmethod
    def _migrate(db: sqlite3.Connection) -> None:
        """Add columns that newer versions need to a database made by an older
        one, so a learner's history survives app updates."""
        have = {row["name"] for row in db.execute("PRAGMA table_info(sessions)")}
        for column, ddl in (("mode", "TEXT NOT NULL DEFAULT 'roleplay'"),
                            ("title", "TEXT NOT NULL DEFAULT ''"),
                            ("updated", "REAL")):
            if column not in have:
                db.execute(f"ALTER TABLE sessions ADD COLUMN {column} {ddl}")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    # ---- profile -------------------------------------------------------
    def get_profile(self) -> dict | None:
        with closing(self._connect()) as db:
            row = db.execute("SELECT data FROM profile WHERE id = 1").fetchone()
        return json.loads(row["data"]) if row else None

    def set_profile(self, profile: dict) -> None:
        with closing(self._connect()) as db, db:
            db.execute("INSERT INTO profile (id, data) VALUES (1, ?) "
                       "ON CONFLICT(id) DO UPDATE SET data = excluded.data", (json.dumps(profile),))

    # ---- sessions ------------------------------------------------------
    def save_session(self, record: dict, summary: dict, title: str = "") -> int:
        with closing(self._connect()) as db, db:
            cur = db.execute(
                "INSERT INTO sessions (started, persona, level, record, summary, mode, title, updated) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (record["started"], record["persona"], record["level"], json.dumps(record),
                 json.dumps(summary), record.get("mode", "roleplay"), title, time.time()))
            session_id = cur.lastrowid
        if self.export_dir is not None:
            from .session import Session
            Session.save(record, self.export_dir)
        return session_id

    def update_session(self, session_id: int, record: dict, summary: dict) -> None:
        """A resumed tutor chat was closed again: replace its record and summary."""
        with closing(self._connect()) as db, db:
            db.execute("UPDATE sessions SET record = ?, summary = ?, level = ?, updated = ? WHERE id = ?",
                       (json.dumps(record), json.dumps(summary), record["level"], time.time(), session_id))
        if self.export_dir is not None:
            from .session import Session
            Session.save(record, self.export_dir)

    def list_conversations(self, limit: int = 50) -> list[dict]:
        """For the sidebar: newest activity first, without the heavy records."""
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT id, started, persona, mode, title, COALESCE(updated, started) AS updated, "
                "json_extract(summary, '$.turns') AS turns FROM sessions "
                "ORDER BY COALESCE(updated, started) DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def list_sessions(self, limit: int | None = None) -> list[dict]:
        """Oldest first (the order progress charts want), with summaries but not full records."""
        sql = "SELECT id, started, persona, level, summary, record FROM sessions ORDER BY started"
        with closing(self._connect()) as db:
            rows = db.execute(sql).fetchall()
        if limit:
            rows = rows[-limit:]
        out = []
        for r in rows:
            record = json.loads(r["record"])
            out.append({"id": r["id"], "started": r["started"], "persona": r["persona"],
                        "mode": record.get("mode", "roleplay"),
                        "level": r["level"], "summary": json.loads(r["summary"]),
                        "duration_s": record.get("duration_s"),
                        "latency": record.get("latency", {}).get("average", {})})
        return out

    def get_session(self, session_id: int) -> dict | None:
        with closing(self._connect()) as db:
            r = db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if not r:
            return None
        return {"id": r["id"], "record": json.loads(r["record"]), "summary": json.loads(r["summary"]),
                "mode": r["mode"], "title": r["title"]}

    # ---- custom scenarios ------------------------------------------------
    def add_scenario(self, data: dict) -> int:
        with closing(self._connect()) as db, db:
            return db.execute("INSERT INTO scenarios (data, created) VALUES (?, ?)",
                              (json.dumps(data), time.time())).lastrowid

    def list_scenarios(self, include_hidden: bool = False) -> list[dict]:
        sql = "SELECT id, data, hidden FROM scenarios" + ("" if include_hidden else " WHERE hidden = 0") + " ORDER BY id"
        with closing(self._connect()) as db:
            return [{"id": r["id"], "hidden": bool(r["hidden"]), **json.loads(r["data"])} for r in db.execute(sql)]

    def hide_scenario(self, scenario_id: int) -> None:
        with closing(self._connect()) as db, db:
            db.execute("UPDATE scenarios SET hidden = 1 WHERE id = ?", (scenario_id,))

    # ---- memory (what the tutor knows about the learner) ----------------
    def add_memories(self, facts: list[str], keep: int = 20) -> None:
        """Store new facts (duplicates ignored) and keep only the newest `keep`,
        so the tutor's prompt stays short."""
        now = time.time()
        with closing(self._connect()) as db, db:
            for i, fact in enumerate(facts):
                # i/1000: same-batch facts keep their order when sorted by time
                db.execute("INSERT OR IGNORE INTO memory (text, created) VALUES (?, ?)", (fact.strip(), now + i / 1000))
            db.execute("DELETE FROM memory WHERE id NOT IN "
                       "(SELECT id FROM memory ORDER BY created DESC LIMIT ?)", (keep,))

    def list_memories(self) -> list[dict]:
        with closing(self._connect()) as db:
            return [dict(r) for r in db.execute("SELECT id, text, created FROM memory ORDER BY created")]

    def delete_memory(self, memory_id: int | None = None) -> None:
        """Forget one fact, or everything when memory_id is None."""
        with closing(self._connect()) as db, db:
            if memory_id is None:
                db.execute("DELETE FROM memory")
            else:
                db.execute("DELETE FROM memory WHERE id = ?", (memory_id,))

    # ---- review cards --------------------------------------------------
    def add_card(self, front: str, data: dict, now: float | None = None) -> bool:
        """Add a card for a flagged span. If it already exists, keep its schedule
        (re-flagging a word must not reset the learner's progress) and only
        refresh the example. Returns True if a new card was created."""
        now = time.time() if now is None else now
        with closing(self._connect()) as db, db:
            cur = db.execute("INSERT OR IGNORE INTO review_cards (front, data, due, created) "
                             "VALUES (?, ?, ?, ?)", (front.lower(), json.dumps(data), now, now))
            if cur.rowcount == 0:
                db.execute("UPDATE review_cards SET data = ? WHERE front = ?",
                           (json.dumps(data), front.lower()))
            return cur.rowcount == 1

    def due_cards(self, now: float | None = None, limit: int = 50) -> list[dict]:
        now = time.time() if now is None else now
        with closing(self._connect()) as db:
            rows = db.execute("SELECT * FROM review_cards WHERE due <= ? ORDER BY due LIMIT ?",
                              (now, limit)).fetchall()
        return [self._card(r) for r in rows]

    def count_due(self, now: float | None = None) -> int:
        now = time.time() if now is None else now
        with closing(self._connect()) as db:
            return db.execute("SELECT COUNT(*) FROM review_cards WHERE due <= ?", (now,)).fetchone()[0]

    def get_card(self, card_id: int) -> dict | None:
        with closing(self._connect()) as db:
            r = db.execute("SELECT * FROM review_cards WHERE id = ?", (card_id,)).fetchone()
        return self._card(r) if r else None

    def update_schedule(self, card_id: int, ease: float, interval: int, reps: int, due: float) -> None:
        with closing(self._connect()) as db, db:
            db.execute("UPDATE review_cards SET ease = ?, interval = ?, reps = ?, due = ? WHERE id = ?",
                       (ease, interval, reps, due, card_id))

    @staticmethod
    def _card(r: sqlite3.Row) -> dict:
        return {"id": r["id"], "front": r["front"], **json.loads(r["data"]), "ease": r["ease"],
                "interval": r["interval"], "reps": r["reps"], "due": r["due"]}
