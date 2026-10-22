import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import streamlit as st
import plotly.express as px
from src import database as db
from src.runtime import get_model

st.set_page_config(page_title="Study Mood Optimizer", page_icon="🎓", layout="wide")
st.title("🎓 AI-Powered Study Mood Optimizer")
st.caption("Multimodal, personalised study planning. Predictions estimate *self-reported study readiness*, not your emotions.")
get_model()
with st.sidebar:
    st.header("Navigation")
    st.markdown("Use the pages above: **Check-in → Analysis → Plan → Evaluation → Progress**")
    if st.button("Load demo history"):
        db.seed_demo(); st.rerun()
d = db.load_sessions()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Sessions", len(d))
c2.metric("Avg usefulness", f"{d.usefulness.mean():.1f}/5" if d.usefulness.notna().any() else "-")
c3.metric("Completion rate", f"{d.completed.mean():.0%}" if d.completed.notna().any() else "-")
c4.metric("Total planned min", int(d.session_minutes.sum()) if len(d) else 0)
st.subheader("Today's plan")
res = st.session_state.get("result")
if res and st.session_state.get("plan"):
    p = st.session_state["plan"]
    st.success(f"**{res['label']}** → {p['technique']}: {p['blocks']} × {p['work_minutes']} min work, {p['break_minutes']} min break")
else:
    st.info("No plan yet. Start with **Study Checkin** in the sidebar.")
if len(d):
    st.subheader("Recent sessions")
    st.dataframe(d.tail(8)[["ts", "task", "label", "technique", "session_minutes", "usefulness", "completed"]], use_container_width=True)
    fig = px.line(d, x="ts", y=["energy", "focus"], markers=True, title="Self-reported energy and focus")
    st.plotly_chart(fig, use_container_width=True)
