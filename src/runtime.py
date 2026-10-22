import streamlit as st
from src.models.train import load_or_train_model

@st.cache_resource(show_spinner="Loading model (first run trains it, ~20 s)...")
def get_model():
    return load_or_train_model()
