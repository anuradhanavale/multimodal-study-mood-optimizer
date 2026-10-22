import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
import pandas as pd
import plotly.express as px
from src.config import LABELS
from src import database as db

from src.runtime import get_model
from src.pipeline import analyze
from src.preprocessing.audio_processor import whisper_available
from src.preprocessing.image_processor import ocr_available
st.title("🎙️ Multimodal Analysis")
ci = st.session_state.get("checkin")
if not ci:
    st.warning("Please complete the Study Check-in first."); st.stop()
st.caption("Voice and image are optional. Processing happens locally; nothing is stored unless you save the session.")
c1, c2 = st.columns(2)
with c1:
    audio = st.audio_input("Voice check-in (optional)") if hasattr(st, "audio_input") else st.file_uploader("Voice check-in (WAV)", type=["wav"])
with c2:
    img = st.file_uploader("Photo of your notes (optional)", type=["png", "jpg", "jpeg"])
st.write(f"Engines: transcription {'✅' if whisper_available() else '⚪ not installed'} · OCR {'✅' if ocr_available() else '⚪ not installed'}")
if st.button("Analyze", type="primary"):
    with st.spinner("Analyzing..."):
        st.session_state["result"] = analyze(get_model(), ci["text"], ci, audio.getvalue() if audio else None, img.getvalue() if img else None)
        st.session_state.pop("plan", None)
res = st.session_state.get("result")
if res:
    for n in res["notes"]: st.warning(n)
    st.metric("Estimated readiness", res["label"], f"confidence {res['confidence']:.0%}")
    st.write("Inputs used:", ", ".join(res["used"]))
    if res["transcript"]: st.write("**Transcript:**", res["transcript"])
    if res["ocr_text"]: st.text_area("OCR text", res["ocr_text"], height=120)
    if res["audio_features"]: st.json(res["audio_features"])
    st.page_link("pages/3_Study_Plan.py", label="Go to Study Plan →")
