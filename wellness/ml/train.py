"""
Train the student-stress text classifier.

Pipeline: TF-IDF (unigrams+bigrams) -> Logistic Regression (multinomial).
- Stratified 80/20 train/test split (fixed seed for reproducibility).
- Reports accuracy, precision / recall / F1 (macro + per class) and the
  confusion matrix.
- Saves vectorizer.pkl, model.pkl and metrics.json into ml/models/.

Run:  python train.py
"""

import json
import pickle
import random
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.pipeline import Pipeline

HERE = Path(__file__).parent
DATA_PATH = HERE / "data" / "stress_dataset.csv"
MODELS_DIR = HERE / "models"
LABEL_NAMES = ["low", "moderate", "high"]


def clean_text(text: str) -> str:
    """Light, label-safe preprocessing for short student-written text."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)      # urls
    text = re.sub(r"[^a-z\s']", " ", text)           # keep letters/apostrophes
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    df = pd.read_csv(DATA_PATH)
    df["clean"] = df["text"].astype(str).map(clean_text)
    df = df[df["clean"].str.len() > 0].reset_index(drop=True)
    print(f"Dataset: {len(df)} samples")

    # Group-aware, class-stratified split: augmented variants of the same base
    # example (same `group`) never land on both sides of the split. This
    # avoids the optimistic bias of near-duplicate leakage and gives honest
    # metrics.
    rng = random.Random(42)
    test_idx = []
    for label in sorted(df["label"].unique()):
        groups = df.loc[df["label"] == label, "group"].unique().tolist()
        rng.shuffle(groups)
        n_test = max(1, int(round(len(groups) * 0.2)))
        test_groups = set(groups[:n_test])
        test_idx.extend(df.loc[df["group"].isin(test_groups)].index.tolist())
    test_mask = df.index.isin(test_idx)
    train_df, test_df = df[~test_mask], df[test_mask]
    X_train, y_train = train_df["clean"], train_df["label"]
    X_test, y_test = test_df["clean"], test_df["label"]
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")
    print("Test class counts:", test_df["label"].value_counts().to_dict())

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            min_df=2,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            C=2.0,
            # lbfgs solver defaults to multinomial for >2 classes
            class_weight="balanced",
            random_state=42,
        )),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=LABEL_NAMES,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    print(f"\nAccuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred) [low, moderate, high]:")
    print(cm)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "vectorizer.pkl", "wb") as f:
        pickle.dump(pipeline.named_steps["tfidf"], f)
    with open(MODELS_DIR / "model.pkl", "wb") as f:
        pickle.dump(pipeline.named_steps["clf"], f)

    metrics = {
        "accuracy": round(float(acc), 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": LABEL_NAMES,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "model": "TF-IDF(1,2-gram) + LogisticRegression(multinomial, balanced)",
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model, vectorizer and metrics.json -> {MODELS_DIR}")


if __name__ == "__main__":
    main()
