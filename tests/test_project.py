import io, os, sys, wave
from pathlib import Path
import numpy as np, pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import database as db
from src.data_gen import generate
from src.models.train import train_and_evaluate
from src.models.fusion_model import ReadinessModel
from src.pipeline import analyze
from src.preprocessing.audio_processor import acoustic_features
from src.preprocessing.image_processor import image_features, ocr_text
from src.recommendations.planner import recommend

CI = dict(energy=2, focus=2, difficulty=4, minutes_available=30, sleep_hours=5.0, recent_completion_rate=0.7)

@pytest.fixture(scope="module")
def trained():
    return train_and_evaluate(n_users=40, per_user=10, save=False)

def wav_bytes(sec=2, sr=16000):
    t = np.linspace(0, sec, sr * sec)
    x = (np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 1.5 * t) > 0) * 12000).astype(np.int16)
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(x.tobytes())
    return b.getvalue()

def test_data_gen():
    d = generate(5, 4); assert len(d) == 20 and set(d.label) <= {0, 1, 2}

def test_train_and_metrics(trained):
    res, model = trained
    assert set(res["comparison"]) >= {"Text only", "Structured only", "Multimodal fusion"}
    assert res["comparison"]["Multimodal fusion"]["macro_f1"] > 0.5
    assert res["comparison"]["Multimodal fusion"]["macro_f1"] >= res["comparison"]["Text only"]["macro_f1"]

def test_missing_modalities(trained):
    _, m = trained
    import pandas as pd
    df = pd.DataFrame([{**CI, "text": ""}])
    p = m.predict_proba(df); assert p.shape == (1, 3) and abs(p.sum() - 1) < 1e-6
    assert m.present_groups(df) == ["struct"]

def test_audio_features():
    f = acoustic_features(wav_bytes()); assert f and f["a_duration"] > 1.9 and 0 <= f["a_pause_ratio"] <= 1
    assert acoustic_features(b"not audio") is None

def test_image_fallbacks():
    assert ocr_text(b"junk") is None or isinstance(ocr_text(b"junk"), str)
    assert image_features(None) is None and image_features("the theorem proof")["i_difficulty_terms"] == 2

def test_pipeline_all_inputs(trained):
    r = analyze(trained[1], "I'm tired and this chapter is hard", CI, wav_bytes(), b"fakeimage")
    assert r["label"] and abs(sum(r["proba"]) - 1) < 1e-6 and "audio" in r["used"] and "struct" in r["used"]

def test_planner_respects_time():
    for mins in (10, 15, 25, 30, 90, 180):
        for lab in range(3):
            p = recommend(lab, [0.4, 0.3, 0.3], {**CI, "minutes_available": mins})
            assert 0 < p["total_minutes"] <= mins and p["low_confidence"]

def test_planner_uses_feedback():
    assert recommend(1, [.2, .6, .2], CI, {"Walk-and-review": 5})["technique"] == "Walk-and-review"

def test_database(tmp_path):
    path = str(tmp_path / "t.db")
    sid = db.save_session({"task": "x", "label": "Ready to study", "technique": "A", "session_minutes": 20}, path)
    db.update_feedback(sid, 5, True, 4, path)
    d = db.load_sessions(path); assert len(d) == 1 and d.usefulness[0] == 5
    assert db.feedback_scores(path) == {"A": 5.0} and db.completion_rate(path) == 1.0
    db.delete_all(path); assert db.load_sessions(path).empty

def test_streamlit_pages(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("STUDY_DB", str(tmp_path / "ui.db"))
    root = Path(__file__).resolve().parents[1]
    from src.models.train import train_and_evaluate as t
    if not (root / "models" / "fusion.joblib").exists(): t()
    for f in ["app.py", "pages/1_Study_Checkin.py", "pages/2_Multimodal_Analysis.py", "pages/3_Study_Plan.py", "pages/4_Model_Evaluation.py", "pages/5_Progress.py"]:
        at = AppTest.from_file(str(root / f), default_timeout=90).run()
        assert not at.exception, f"{f}: {at.exception}"
    # full journey
    at = AppTest.from_file(str(root / "pages/1_Study_Checkin.py"), default_timeout=90).run()
    at.text_area[0].set_value("tired and this is hard"); at.button[0].click().run()
    assert "checkin" in at.session_state
    ci = at.session_state["checkin"]
    at2 = AppTest.from_file(str(root / "pages/3_Study_Plan.py"), default_timeout=90)
    at2.session_state["checkin"] = ci
    from src.models.train import load_or_train_model
    at2.session_state["result"] = analyze(load_or_train_model(), ci["text"], ci)
    at2.run(); assert not at2.exception
    at2.button[0].click().run(); assert not at2.exception and "saved_id" in at2.session_state
