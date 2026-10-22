"""Images of notes: optional OCR (pytesseract) + simple text-derived features."""
import io, re

def ocr_available():
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False

def ocr_text(image_bytes):
    if not image_bytes or not ocr_available():
        return None
    try:
        import pytesseract
        from PIL import Image
        return pytesseract.image_to_string(Image.open(io.BytesIO(image_bytes))).strip() or None
    except Exception:
        return None

HARD = {"theorem", "proof", "derive", "equation", "integral", "algorithm", "complexity", "lemma", "derivation", "formula",
        "hypothesis", "mechanism", "synthesis", "differential", "matrix"}

def image_features(text):
    if not text:
        return None
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {"i_chars": len(text), "i_difficulty_terms": sum(w in HARD for w in words), "i_density": len(text) / 1500.0}
