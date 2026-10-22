"""Voice pipeline, STT/TTS wrappers and storage, all with fakes (no models)."""

import io
import sys
import wave

import numpy as np
import pytest

from coach.errors import Unavailable
from coach.personas import load_personas
from coach.session import Session, average_latency
from coach.storage import Storage
from coach.tts import KokoroTTS, to_wav
from coach.voice import VoicePipeline
from tests.fakes import FakeLLM, FakeSTT, FakeTTS


def make_pipeline(stt=None, tts=None, reply="Oh nice. Where did you go? Tell me!"):
    session = Session(load_personas()["buddy"], FakeLLM(reply), level="B1")
    return VoicePipeline(session, stt, tts), session


def test_voice_turn_events_in_order():
    tts = FakeTTS()
    pipe, session = make_pipeline(FakeSTT("I went to Goa."), tts)
    events = list(pipe.run_turn(audio=b"fake-webm"))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "transcript" and events[0]["text"] == "I went to Goa."
    assert kinds[-1] == "done"
    audio = [e for e in events if e["type"] == "audio"]
    # one audio event per sentence, in order, the first one before the reply ends
    assert [a["text"] for a in audio] == ["Oh nice.", "Where did you go?", "Tell me!"]
    assert [a["index"] for a in audio] == [0, 1, 2]
    assert kinds.index("audio") < max(i for i, k in enumerate(kinds) if k == "text")
    assert tts.spoken[0][1] == "american_female"   # buddy's persona voice


def test_latency_is_measured_and_averaged():
    pipe, session = make_pipeline(FakeSTT(), FakeTTS())
    list(pipe.run_turn(audio=b"x"))
    list(pipe.run_turn(text="Typed answer."))
    voice_t, text_t = session.latencies
    assert {"stt_s", "llm_first_token_s", "tts_first_audio_s", "total_s"} <= set(voice_t)
    assert "stt_s" not in text_t and text_t["mode"] == "text"
    record = session.finish()
    avg = record["latency"]["average"]
    assert avg["stt_s"] == voice_t["stt_s"]          # averaged only over voice turns
    assert "total_s" in avg


def test_average_latency_skips_missing_keys():
    assert average_latency([{"a": 1.0}, {"a": 3.0, "b": 2.0}, {"mode": "text"}]) == {"a": 2.0, "b": 2.0}


def test_works_without_tts_and_rejects_audio_without_stt():
    pipe, _ = make_pipeline(stt=None, tts=None)
    events = list(pipe.run_turn(text="Hello!"))
    assert all(e["wav"] is None for e in events if e["type"] == "audio")
    with pytest.raises(Unavailable):
        list(pipe.run_turn(audio=b"x"))


def test_empty_transcript_skips_the_llm():
    pipe, session = make_pipeline(FakeSTT("  "), FakeTTS())
    events = list(pipe.run_turn(audio=b"x"))
    assert events[-1]["empty"] is True
    assert session.llm.seen == [] and session.turns == []


def test_barge_in_keeps_partial_reply():
    pipe, session = make_pipeline(FakeSTT(), FakeTTS())
    gen = pipe.run_turn(text="Hi.")
    for e in gen:
        if e["type"] == "audio":
            break
    gen.close()   # the learner interrupted after the first sentence
    assert session.messages[-1] == {"role": "assistant", "content": "Oh nice. "}
    assert session.latencies, "interrupted turns still count for latency"


def test_opening_is_spoken():
    tts = FakeTTS()
    pipe, session = make_pipeline(tts=tts)
    ev = pipe.opening()
    assert ev["text"] == session.opening and ev["wav"].startswith(b"RIFF")


def test_to_wav_is_valid_16bit_mono():
    data = to_wav(np.array([0.0, 0.5, -1.5], dtype="float32"), 24000)
    with wave.open(io.BytesIO(data)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, 24000, 3)


def test_kokoro_missing_files_is_unavailable(tmp_path):
    with pytest.raises(Unavailable, match="setup"):
        KokoroTTS({"american_female": "af_heart"}, model_dir=tmp_path)


def test_whisper_missing_library_is_unavailable(monkeypatch):
    from coach.stt import WhisperSTT
    monkeypatch.setitem(sys.modules, "faster_whisper", None)   # makes the import fail
    with pytest.raises(Unavailable, match="Voice off"):
        WhisperSTT("tiny")


def test_storage_profile_sessions_and_cards(tmp_path):
    db = Storage(tmp_path / "g.db", export_dir=tmp_path / "export")
    assert db.get_profile() is None
    db.set_profile({"name": "Asha", "goal": "job"})
    db.set_profile({"name": "Asha", "goal": "placements"})
    assert db.get_profile()["goal"] == "placements"

    _, session = make_pipeline(tts=FakeTTS())
    list(VoicePipeline(session, None, None).run_turn(text="It was very good."))
    record = session.finish()
    sid = db.save_session(record, {"turns": 1})
    assert db.get_session(sid)["record"]["persona"] == "buddy"
    assert [s["id"] for s in db.list_sessions()] == [sid]
    assert len(list((tmp_path / "export").glob("*.json"))) == 1   # JSON export kept

    assert db.add_card("Very Good", {"suggestions": ["excellent"]}, now=100) is True
    assert db.add_card("very good", {"suggestions": ["superb"]}, now=200) is False
    db.update_schedule(db.due_cards(now=100)[0]["id"], 2.6, 1, 1, due=500)
    assert db.count_due(now=400) == 0 and db.count_due(now=600) == 1
    card = db.due_cards(now=600)[0]
    assert card["front"] == "very good" and card["suggestions"] == ["superb"] and card["reps"] == 1
