import numpy as np, pandas as pd
from src.config import LABELS
from src.preprocessing.audio_processor import acoustic_features, transcribe, whisper_available
from src.preprocessing.image_processor import image_features, ocr_available, ocr_text

def analyze(model, text, checkin, audio_bytes=None, image_bytes=None):
    notes = []
    transcript = transcribe(audio_bytes) if audio_bytes else None
    afeat = acoustic_features(audio_bytes) if audio_bytes else None
    if audio_bytes and not whisper_available(): notes.append("faster-whisper not installed: audio used for acoustic features only (no transcript).")
    if audio_bytes and afeat is None: notes.append("Audio could not be parsed (WAV expected) or was too short; audio ignored.")
    ocr = ocr_text(image_bytes) if image_bytes else None
    ifeat = image_features(ocr)
    if image_bytes and not ocr_available(): notes.append("OCR engine not installed: image not used.")
    elif image_bytes and ifeat is None: notes.append("No text found in the image; image ignored.")
    full_text = " ".join(x for x in [text or "", transcript or ""] if x).strip()
    row = {"text": full_text, **{k: checkin[k] for k in ["energy", "focus", "difficulty", "minutes_available", "sleep_hours", "recent_completion_rate"]}}
    row.update(afeat or {}); row.update(ifeat or {})
    df = pd.DataFrame([row])
    proba = model.predict_proba(df)[0]
    k = int(np.argmax(proba))
    return {"label_idx": k, "label": LABELS[k], "confidence": float(proba[k]), "proba": proba.tolist(),
            "explanation": model.explain(df), "used": model.present_groups(df), "transcript": transcript, "ocr_text": ocr,
            "audio_features": afeat, "image_features": ifeat, "notes": notes}
