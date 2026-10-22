import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
import pandas as pd
import plotly.express as px
from src.config import LABELS
from src import database as db

import json
from src.config import MODEL_DIR
from src.models.train import train_and_evaluate
st.title("🧪 AI Evaluation Lab")
st.warning("Metrics below come from a **synthetic** development dataset. Replace it with your consented dataset before making research claims.")
if st.button("Retrain & re-evaluate"):
    with st.spinner("Training..."):
        train_and_evaluate(); st.cache_resource.clear()
p = MODEL_DIR / "metrics.json"
if not p.exists():
    with st.spinner("Training first model..."): train_and_evaluate()
m = json.loads(p.read_text())
st.caption(f"Train rows {m['n_train']} · test rows {m['n_test']} ({m['n_test_users']} held-out participants, split by participant) · median latency {m['median_latency_ms']:.1f} ms")
t = pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk != "confusion"} for k, v in m["comparison"].items()}).T
st.dataframe(t.style.format("{:.3f}"), use_container_width=True)
st.plotly_chart(px.bar(t.reset_index(), x="index", y="macro_f1", title="Macro F1: unimodal vs multimodal", labels={"index": ""}), use_container_width=True)
r = pd.DataFrame(m["robustness"]).reset_index().melt("index", var_name="model", value_name="macro_f1")
st.plotly_chart(px.bar(r, x="index", y="macro_f1", color="model", barmode="group", title="Missing-modality robustness", labels={"index": "Condition"}), use_container_width=True)
pick = st.selectbox("Confusion matrix for", list(m["comparison"]), index=len(m["comparison"]) - 1)
st.plotly_chart(px.imshow(m["comparison"][pick]["confusion"], x=LABELS, y=LABELS, text_auto=True, labels=dict(x="Predicted", y="True")), use_container_width=True)
