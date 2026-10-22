import os, sqlite3
from datetime import datetime, timedelta
import numpy as np, pandas as pd
from src.config import DB_PATH, LABELS

COLS = ["ts", "task", "text", "energy", "focus", "difficulty", "minutes", "sleep", "label", "confidence", "technique",
        "session_minutes", "used", "usefulness", "completed", "focus_after", "is_demo"]

def _conn(path=None):
    p = path or os.environ.get("STUDY_DB") or str(DB_PATH)
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    c = sqlite3.connect(p); c.row_factory = sqlite3.Row
    c.execute("""CREATE TABLE IF NOT EXISTS sessions(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, task TEXT, text TEXT,
        energy INT, focus INT, difficulty INT, minutes INT, sleep REAL, label TEXT, confidence REAL, technique TEXT,
        session_minutes INT, used TEXT, usefulness INT, completed INT, focus_after INT, is_demo INT DEFAULT 0)""")
    return c

def save_session(rec, path=None):
    rec = {k: rec.get(k) for k in COLS}
    rec["ts"] = rec["ts"] or datetime.now().isoformat(timespec="seconds"); rec["is_demo"] = rec["is_demo"] or 0
    with _conn(path) as c:
        cur = c.execute(f"INSERT INTO sessions({','.join(COLS)}) VALUES({','.join('?' * len(COLS))})", [rec[k] for k in COLS])
        return cur.lastrowid

def update_feedback(sid, usefulness, completed, focus_after, path=None):
    with _conn(path) as c:
        c.execute("UPDATE sessions SET usefulness=?, completed=?, focus_after=? WHERE id=?", (usefulness, int(completed), focus_after, sid))

def load_sessions(path=None):
    with _conn(path) as c:
        return pd.read_sql_query("SELECT * FROM sessions ORDER BY ts", c)

def delete_all(path=None):
    with _conn(path) as c:
        c.execute("DELETE FROM sessions")

def feedback_scores(path=None):
    d = load_sessions(path).dropna(subset=["usefulness"])
    return d.groupby("technique").usefulness.mean().to_dict() if len(d) else {}

def completion_rate(path=None, default=0.7):
    d = load_sessions(path).dropna(subset=["completed"])
    return float(d.completed.mean()) if len(d) else default

def seed_demo(n=14, path=None):
    rng = np.random.default_rng(7)
    for i in range(n):
        lab = int(rng.choice(3, p=[0.4, 0.35, 0.25]))
        save_session({"ts": (datetime.now() - timedelta(days=n - i)).isoformat(timespec="seconds"), "task": "Demo session",
                      "text": "demo", "energy": int(rng.integers(1, 6)), "focus": int(rng.integers(1, 6)), "difficulty": int(rng.integers(1, 6)),
                      "minutes": 30, "sleep": 7.0, "label": LABELS[lab], "confidence": float(rng.uniform(0.5, 0.9)),
                      "technique": "Pomodoro with early break", "session_minutes": int(rng.choice([15, 20, 30])), "used": "struct,text",
                      "usefulness": int(rng.integers(3, 6)), "completed": int(rng.random() < 0.75), "focus_after": int(rng.integers(2, 6)),
                      "is_demo": 1}, path)
