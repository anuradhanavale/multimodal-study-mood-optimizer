"""Synthetic study-session generator for development/demo ONLY.
Real claims need your own consented dataset (see README)."""
import numpy as np, pandas as pd
from src.config import LABELS

HIGH = ["I feel rested and ready to dive in", "Feeling focused and motivated today", "I'm energized and confident about studying"]
MID = ["I'm okay but a bit distracted", "Feeling fine though I could use a short break soon", "A little bored and unfocused but managing"]
LOW = ["I'm exhausted and drained", "Feeling tired, stressed and overwhelmed", "So sleepy and I can't concentrate, I'm stuck"]
HARDP = ["this chapter is really hard", "the material is difficult and confusing", "I'm lost on these concepts"]
EASYP = ["the material looks easy", "the topic is clear and simple"]

def _label(readiness, diff):
    if readiness > 3.2:
        return 0
    if readiness < 2.5 or (diff >= 4 and readiness < 3.0):
        return 2
    return 1

def generate(n_users=80, per_user=12, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        bias, comp = rng.normal(0, 0.3), rng.uniform(0.4, 0.95)
        for _ in range(per_user):
            energy = int(np.clip(round(rng.normal(3, 1.1)), 1, 5))
            focus = int(np.clip(round(0.5 * energy + rng.normal(1.5, 1.0)), 1, 5))
            diff = int(rng.integers(1, 6))
            sleep = float(np.clip(rng.normal(7, 1.3), 3, 10))
            mins = int(rng.choice([15, 20, 30, 45, 60, 90]))
            ready = (energy + focus) / 2 - 0.45 * (diff - 3) + 0.2 * (sleep - 7) + bias + rng.normal(0, 0.35)
            felt = ready + rng.normal(0, 0.5)
            pool = HIGH if felt > 3.2 else MID if felt > 2.5 else LOW
            text = rng.choice(pool) + (", " + rng.choice(HARDP if diff >= 4 else EASYP) if diff != 3 else "")
            if rng.random() < 0.1:
                text = ""
            row = dict(user_id=u, text=text, energy=energy, focus=focus, difficulty=diff, minutes_available=mins,
                       sleep_hours=sleep, recent_completion_rate=float(np.clip(comp + rng.normal(0, 0.05), 0, 1)),
                       a_duration=np.nan, a_rms=np.nan, a_zcr=np.nan, a_pause_ratio=np.nan,
                       i_chars=np.nan, i_difficulty_terms=np.nan, i_density=np.nan, label=_label(ready, diff))
            if rng.random() < 0.5:
                row.update(a_duration=rng.uniform(4, 20), a_rms=float(np.clip(0.08 + 0.02 * (felt - 3) + rng.normal(0, 0.015), 0.01, 0.3)),
                           a_zcr=float(np.clip(0.08 + rng.normal(0, 0.02), 0.01, 0.3)),
                           a_pause_ratio=float(np.clip(0.35 - 0.06 * (felt - 3) + rng.normal(0, 0.08), 0.05, 0.9)))
            if rng.random() < 0.4:
                chars = rng.uniform(200, 1500)
                row.update(i_chars=chars, i_difficulty_terms=float(rng.poisson(max(0.5, diff))), i_density=chars / 1500 + rng.normal(0, 0.05))
            rows.append(row)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    d = generate(); print(d.shape, d.label.map(dict(enumerate(LABELS))).value_counts().to_dict())
