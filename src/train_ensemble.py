# Import modules
from __future__ import annotations
import os, json, joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from src.datasets import load_text_dataset
from src.logger import logger

# Train the model with defined parameters, revert to these if default
def train_ensemble(
    data_path: str | None,
    tfidf_max_features: int = 20000,
    tfidf_ngram_max: int = 2,
    tfidf_min_df: int = 2,
    rf_estimators: int = 300,
):
    # Log to serial
    logger.info("[ENS] Loading dataset...")
    texts, y = load_text_dataset(data_path)

    # Split data between test/train
    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, y, test_size=0.25, stratify=y, random_state=42
    )

    vect = TfidfVectorizer( # Create TFIDF
        max_features=tfidf_max_features,
        ngram_range=(1, tfidf_ngram_max),
        min_df=tfidf_min_df,
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
    )

    # Transform TFIDFs
    Xtr = vect.fit_transform(X_tr)
    Xte = vect.transform(X_te)

    # Set RF Count
    rf = RandomForestClassifier(
        n_estimators=rf_estimators,
        n_jobs=-1,
        random_state=42
    )
    
    # Log to serial and fit training data
    logger.info("[ENS] Training RF...")
    rf.fit(Xtr, y_tr)

    # Print metric scores
    y_pred = rf.predict(Xte)
    print(classification_report(y_te, y_pred, digits=4, zero_division=0))

    # Save models
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(rf, "saved_models/ensemble_rf.joblib")
    joblib.dump(vect, "saved_models/ensemble_tfidf.joblib")
    with open("saved_models/ensemble_meta.json", "w") as f:
        json.dump(
            dict(tfidf_max_features=tfidf_max_features,
                 tfidf_ngram_max=tfidf_ngram_max,
                 tfidf_min_df=tfidf_min_df,
                 rf_estimators=rf_estimators),
            f, indent=2
        )
    logger.info("[ENS] Saved RF + TF-IDF.")
