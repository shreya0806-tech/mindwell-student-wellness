"""
Load the trained stress classifier and expose a simple predict() API.

Artifacts (produced by wellness/ml/train.py):
    wellness/ml/models/vectorizer.pkl
    wellness/ml/models/model.pkl
    wellness/ml/models/metrics.json

The model is loaded once and cached. predict() returns the label
("low" | "moderate" | "high"), the per-class probabilities, and a
confidence value. Confidence is the max class probability, shown in the UI
only as a soft indicator — never as a medical probability.
"""

import json
import pickle
import re
from pathlib import Path

LABELS = ["low", "moderate", "high"]
MODELS_DIR = Path(__file__).resolve().parent / "models"

_model = None
_vectorizer = None
_metrics = None


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_model():
    """Load (vectorizer, model) once; raise a clear error if files are missing."""
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        vec_path = MODELS_DIR / "vectorizer.pkl"
        model_path = MODELS_DIR / "model.pkl"
        if not vec_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                "Trained model files not found. Run "
                "`python wellness/ml/train.py` first (see README)."
            )
        with open(vec_path, "rb") as f:
            _vectorizer = pickle.load(f)
        with open(model_path, "rb") as f:
            _model = pickle.load(f)
    return _vectorizer, _model


def get_metrics():
    global _metrics
    if _metrics is None:
        path = MODELS_DIR / "metrics.json"
        _metrics = json.loads(path.read_text()) if path.exists() else {}
    return _metrics


def predict(text: str) -> dict:
    """Classify student-written text into a stress category."""
    cleaned = clean_text(text or "")
    if not cleaned:
        raise ValueError("Please write a few words about how you feel first.")
    vectorizer, model = get_model()
    X = vectorizer.transform([cleaned])
    proba = model.predict_proba(X)[0]
    idx = int(proba.argmax())
    return {
        "label": LABELS[idx],
        "confidence": round(float(proba[idx]), 3),
        "probabilities": {LABELS[i]: round(float(p), 3) for i, p in enumerate(proba)},
    }
