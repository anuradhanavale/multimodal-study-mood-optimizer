# AI-Powered Study Mood Optimizer
Multimodal personalised study planner (Streamlit + scikit-learn). Estimates **self-reported study readiness**
(Ready / Needs a short break / Needs a lighter task) from text, voice, note images and check-in data, then builds a study plan.

## Run
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m src.models.train                            # optional: pre-trained model is included
streamlit run app.py
```
Optional engines: `pip install -r requirements-optional.txt` (faster-whisper for transcripts; pytesseract + Tesseract binary for OCR).
Without them the app still works: audio contributes acoustic features, images are skipped, and the UI says so.

## Test
`python -m pytest -q` (unit tests + Streamlit AppTest of every page)

## Architecture
Late fusion: struct encoder (6 check-in features) · text encoder (TF-IDF → SVD + lexicon) · audio features (duration, RMS, ZCR, pause ratio)
· image features (OCR-derived) → concatenation + presence flags → MLP. Trained with **random modality dropout**, so any input can be missing.
Explainability = feature-group ablation. Recommendations come from a rule-based planner that uses prediction, time limit, difficulty and your past feedback (not an LLM).

## IMPORTANT: data honesty
The included model is trained on **synthetic data** (`src/data_gen.py`) so the full system can be demonstrated and tested.
Its metrics prove the *pipeline works*, not that it predicts real students. For your report:
1. Collect an opt-in dataset (consent, secure storage, ethics approval) with the pre/post questions in the project plan.
2. Save it as a CSV with columns from `src/config.py` + `user_id`, `text`, `label`, and swap it into `train_and_evaluate()` in `src/models/train.py`.
3. Keep the participant-level split, report measured macro-F1 vs the text-only baseline, and discuss failures.
No facial emotion recognition is used, by design.

## Structure
`app.py` dashboard · `pages/` check-in, analysis, plan, evaluation lab, progress · `src/` preprocessing, models, recommendations, evaluation, database · `tests/`
