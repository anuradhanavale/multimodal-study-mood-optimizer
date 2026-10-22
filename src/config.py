from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DB_PATH = DATA_DIR / "study.db"
LABELS = ["Ready to study", "Needs a short break", "Needs a lighter task"]
STRUCT_COLS = ["energy", "focus", "difficulty", "minutes_available", "sleep_hours", "recent_completion_rate"]
AUDIO_COLS = ["a_duration", "a_rms", "a_zcr", "a_pause_ratio"]
IMAGE_COLS = ["i_chars", "i_difficulty_terms", "i_density"]
