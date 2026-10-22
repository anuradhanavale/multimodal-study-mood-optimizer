import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
import pandas as pd
import plotly.express as px
from src.config import LABELS
from src import database as db

st.title("📈 Progress & Feedback")
d = db.load_sessions()
if d.empty:
    st.info("No sessions yet. Save a session from the Study Plan page, or use 'Load demo history' on the Dashboard.")
else:
    st.plotly_chart(px.bar(d, x="ts", y="session_minutes", color="label", title="Planned study minutes"), use_container_width=True)
    f = d.dropna(subset=["usefulness"])
    if len(f):
        st.plotly_chart(px.line(f, x="ts", y=["usefulness", "focus", "focus_after"], markers=True, title="Usefulness and focus (before / after)"), use_container_width=True)
        st.metric("Task completion", f"{f.completed.mean():.0%}")
    st.dataframe(d, use_container_width=True)
st.divider()
st.subheader("Privacy")
st.caption("Audio and images are never written to disk by this app; only derived values you choose to save are stored locally in SQLite.")
if st.button("Delete ALL stored sessions"):
    db.delete_all(); st.success("Deleted."); st.rerun()
