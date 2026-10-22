"""Audio: optional transcription (faster-whisper) + lightweight acoustic features (numpy only)."""
import io, os, tempfile, wave
import numpy as np

def whisper_available():
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False

def transcribe(wav_bytes):
    """Return transcript text or None if no ASR engine is installed / transcription fails."""
    if not wav_bytes or not whisper_available():
        return None
    try:
        from faster_whisper import WhisperModel
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes); path = f.name
        try:
            segs, _ = WhisperModel("tiny", device="cpu", compute_type="int8").transcribe(path)
            return " ".join(s.text.strip() for s in segs).strip() or None
        finally:
            os.unlink(path)
    except Exception:
        return None

def acoustic_features(wav_bytes):
    """Return dict(a_duration, a_rms, a_zcr, a_pause_ratio) or None if audio can't be parsed."""
    try:
        with wave.open(io.BytesIO(wav_bytes)) as w:
            n, sr, ch, sw = w.getnframes(), w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(n)
        dt = {1: np.uint8, 2: np.int16, 4: np.int32}[sw]
        x = np.frombuffer(raw, dtype=dt).astype(float)
        if sw == 1:
            x -= 128
        x /= float(2 ** (8 * sw - 1))
        if ch > 1:
            x = x[: len(x) // ch * ch].reshape(-1, ch).mean(1)
        if len(x) < sr * 0.2:
            return None
        fl = int(sr * 0.05)
        fr = x[: len(x) // fl * fl].reshape(-1, fl)
        env = np.sqrt((fr ** 2).mean(1))
        pause = float((env < 0.1 * env.max()).mean()) if env.max() > 0 else 1.0
        zcr = float(np.abs(np.diff(np.sign(x))).mean() / 2)
        return {"a_duration": len(x) / sr, "a_rms": float(np.sqrt((x ** 2).mean())), "a_zcr": zcr, "a_pause_ratio": pause}
    except Exception:
        return None
