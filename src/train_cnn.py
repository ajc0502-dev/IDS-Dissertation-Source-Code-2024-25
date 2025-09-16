# Import modules
from __future__ import annotations
import os, json, joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.datasets import load_text_dataset
from src.models.cnn import CNNIDS
from src.logger import logger

# Train the model with defined parameters, revert to these if default
def train_cnn(
    data_path: str | None,
    max_len: int = 100,
    vocab_size: int = 20000,
    epochs: int = 10,
    batch_size: int = 64,
    dropout: float = 0.3
):
    # Log to serial
    logger.info("[CNN] Loading dataset...")
    texts, y = load_text_dataset(data_path)

    # Split data between test/train
    X_tr, X_te, y_tr, y_te = train_test_split(texts, y, test_size=0.25, stratify=y, random_state=42)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=42)

    # Fit data and log to serial
    model = CNNIDS(max_len=max_len, vocab_size=vocab_size, dropout_rate=dropout)
    logger.info("[CNN] Fitting...")
    history = model.fit(
        X_tr, y_tr,
        X_va, y_va,
        epochs=epochs,
        batch_size=batch_size
    )

    # MAKE SURE TO EVALUATE BASED ON THRESHOLDS, log to serial
    logger.info("[CNN] Evaluating...")
    model.evaluate(X_te, y_te)

    # Save model
    os.makedirs("saved_models", exist_ok=True)
    model_path = "saved_models/cnn_ids.keras"
    tok_path = "saved_models/cnn_tokenizer.joblib"
    meta_path = "saved_models/cnn_meta.json"
    model.save(model_path, tok_path, meta_path)
    logger.info("[CNN] Saved CNN model, tokenizer, and meta.")
