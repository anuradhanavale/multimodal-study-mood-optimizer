import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
import pandas as pd
import plotly.express as px
from src.config import LABELS
from src import database as db

from src.recommendations.planner import recommend
st.title("📅 Personalised Study Plan")
res, ci = st.session_state.get("result"), st.session_state.get("checkin")
if not res or not ci:
    st.warning("Run the analysis first."); st.stop()
plan = recommend(res["label_idx"], res["proba"], ci, db.feedback_scores())
st.session_state["plan"] = plan
st.header(res["label"])
if plan["low_confidence"]: st.warning("Low confidence prediction - treat this as a gentle suggestion.")
st.success(f"**{plan['technique']}** - {plan['description']}")
c1, c2, c3 = st.columns(3)
c1.metric("Work block", f"{plan['work_minutes']} min"); c2.metric("Break", f"{plan['break_minutes']} min"); c3.metric("Blocks", plan["blocks"])
left, right = st.columns(2)
with left:
    st.plotly_chart(px.bar(x=LABELS, y=res["proba"], labels={"x": "", "y": "Probability"}, title="Class probabilities"), use_container_width=True)
with right:
    ex = res["explanation"]
    st.plotly_chart(px.bar(x=list(ex.keys()), y=list(ex.values()), labels={"x": "Input group", "y": "Change in confidence if removed"},
                           title="Which inputs mattered (ablation)"), use_container_width=True)
st.subheader("Why?"); [st.write("• " + r) for r in plan["reasons"]]
if "saved_id" not in st.session_state:
    if st.button("Save this session"):
        st.session_state["saved_id"] = db.save_session({"task": ci["task"], "text": ci["text"], "energy": ci["energy"], "focus": ci["focus"],
            "difficulty": ci["difficulty"], "minutes": ci["minutes_available"], "sleep": ci["sleep_hours"], "label": res["label"],
            "confidence": res["confidence"], "technique": plan["technique"], "session_minutes": plan["total_minutes"], "used": ",".join(res["used"])})
        st.rerun()
else:
    st.info("Session saved. After studying, give feedback:")
    with st.form("fb"):
        u = st.slider("Was the recommendation useful?", 1, 5, 4); done = st.checkbox("I completed my planned task")
        fa = st.slider("Focus after the session", 1, 5, 3)
        if st.form_submit_button("Submit feedback"):
            db.update_feedback(st.session_state["saved_id"], u, done, fa); st.success("Thanks! Future plans will use this feedback.")
            del st.session_state["saved_id"]
