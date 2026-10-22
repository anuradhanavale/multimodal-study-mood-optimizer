import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
import pandas as pd
import plotly.express as px
from src.config import LABELS
from src import database as db

st.title("📝 Study Check-in")
with st.form("checkin_form"):
    text = st.text_area("How are you feeling about studying? (optional)", placeholder="I'm tired and finding this chapter difficult")
    task = st.text_input("What do you want to study?", "Chapter review")
    c1, c2, c3 = st.columns(3)
    energy = c1.slider("Energy", 1, 5, 3); focus = c2.slider("Focus", 1, 5, 3); diff = c3.slider("Task difficulty", 1, 5, 3)
    c4, c5 = st.columns(2)
    minutes = c4.number_input("Minutes available", 10, 180, 30, 5)
    sleep = c5.slider("Hours slept last night", 0.0, 12.0, 7.0, 0.5)
    if st.form_submit_button("Save check-in"):
        st.session_state["checkin"] = dict(text=text, task=task, energy=energy, focus=focus, difficulty=diff,
            minutes_available=int(minutes), sleep_hours=float(sleep), recent_completion_rate=db.completion_rate())
        st.session_state.pop("result", None); st.session_state.pop("plan", None)
        st.success("Check-in saved. Continue to **Multimodal Analysis**.")
