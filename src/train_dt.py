# Import modules
from __future__ import annotations
import os, json, joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from src.datasets import load_text_dataset
from src.logger import logger

# Train the model with defined parameters, revert to these if default
def train_decision_tree(
    data_path: str | None,
    max_depth: int = 5,
    tfidf_max_features: int = 20000,
    tfidf_ngram_max: int = 2,
    tfidf_min_df: int = 2,
):
     # Split data between test/train
    logger.info("[DT] Loading dataset...")
    texts, y = load_text_dataset(data_path)

    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, y, test_size=0.25, stratify=y, random_state=42
    )

    vect = TfidfVectorizer(
        max_features=tfidf_max_features,
        ngram_range=(1, tfidf_ngram_max),
        min_df=tfidf_min_df,
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
    )

    # Transform TFIDFs
    Xtr = vect.fit_transform(X_tr)
    Xte = vect.transform(X_te)

    # Log to serial and fit training data
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    logger.info("[DT] Training...")
    clf.fit(Xtr, y_tr)

    # Predict DT and print metric
    y_pred = clf.predict(Xte)
    print(classification_report(y_te, y_pred, digits=4, zero_division=0))

    # Save model
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(clf, "saved_models/dt_model.joblib")
    joblib.dump(vect, "saved_models/dt_tfidf.joblib")
    with open("saved_models/dt_meta.json", "w") as f:
        json.dump(
            dict(max_depth=max_depth, tfidf_max_features=tfidf_max_features,
                 tfidf_ngram_max=tfidf_ngram_max, tfidf_min_df=tfidf_min_df),
            f, indent=2
        )
    logger.info("[DT] Saved model and vectorizer.")
