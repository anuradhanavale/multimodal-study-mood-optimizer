import json, statistics, time
import joblib
from sklearn.model_selection import GroupShuffleSplit
from src.config import DATA_DIR, MODEL_DIR
from src.data_gen import generate
from src.evaluation.metrics import evaluate
from src.models.fusion_model import ReadinessModel

CONDITIONS = {"As collected": (), "No audio": ("audio",), "No image": ("image",), "Check-in + text only": ("audio", "image"),
              "No text": ("text",), "Check-in only": ("text", "audio", "image")}

def train_and_evaluate(n_users=80, per_user=12, seed=42, save=True):
    df = generate(n_users, per_user, seed)
    # Split by participant, never by row, to avoid identity leakage.
    tr_i, te_i = next(GroupShuffleSplit(1, test_size=0.25, random_state=seed).split(df, groups=df.user_id))
    tr, te = df.iloc[tr_i].reset_index(drop=True), df.iloc[te_i].reset_index(drop=True)
    variants = {"Text only": ReadinessModel(("text",), False, seed), "Structured only": ReadinessModel(("struct",), False, seed),
                "Text + Audio": ReadinessModel(("text", "audio"), False, seed),
                "Multimodal fusion (no dropout)": ReadinessModel(dropout=False, seed=seed),
                "Multimodal fusion": ReadinessModel(dropout=True, seed=seed)}
    res = {"comparison": {}, "robustness": {}, "n_train": len(tr), "n_test": len(te), "n_test_users": int(te.user_id.nunique()),
           "data_source": "synthetic (development only)"}
    for name, m in variants.items():
        m.fit(tr, tr.label)
        res["comparison"][name] = evaluate(te.label, m.predict_proba(te))
    for name in ("Multimodal fusion (no dropout)", "Multimodal fusion"):
        res["robustness"][name] = {c: evaluate(te.label, variants[name].predict_proba(te, d))["macro_f1"] for c, d in CONDITIONS.items()}
    fusion, one, times = variants["Multimodal fusion"], te.iloc[[0]], []
    for _ in range(30):
        t = time.perf_counter(); fusion.predict_proba(one); times.append((time.perf_counter() - t) * 1000)
    res["median_latency_ms"] = statistics.median(times)
    if save:
        MODEL_DIR.mkdir(exist_ok=True); DATA_DIR.mkdir(exist_ok=True)
        joblib.dump(fusion, MODEL_DIR / "fusion.joblib")
        (MODEL_DIR / "metrics.json").write_text(json.dumps(res, indent=2))
        df.to_csv(DATA_DIR / "processed" / "synthetic_sessions.csv", index=False)
    return res, fusion

def load_or_train_model():
    p = MODEL_DIR / "fusion.joblib"
    if p.exists():
        try:
            return joblib.load(p)
        except Exception:
            pass
    return train_and_evaluate()[1]

if __name__ == "__main__":
    r, _ = train_and_evaluate()
    for k, v in r["comparison"].items():
        print(f"{k:34s} macroF1={v['macro_f1']:.3f} balAcc={v['balanced_accuracy']:.3f} brier={v['brier']:.3f}")
    print("robustness:", json.dumps(r["robustness"]["Multimodal fusion"], indent=1)); print("latency ms:", round(r["median_latency_ms"], 1))
