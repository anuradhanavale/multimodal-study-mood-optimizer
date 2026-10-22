import re
POS = {"ready", "rested", "focused", "motivated", "energized", "confident", "good", "great", "clear", "calm", "excited", "fine", "easy"}
NEG = {"tired", "exhausted", "sleepy", "overwhelmed", "stressed", "anxious", "distracted", "confused", "hard", "difficult",
       "bored", "stuck", "drained", "lost", "worried"}

def clean(t):
    return re.sub(r"\s+", " ", (t or "").strip().lower())

def lexicon_features(t):
    words = re.findall(r"[a-z']+", clean(t))
    return [sum(w in POS for w in words), sum(w in NEG for w in words)]
