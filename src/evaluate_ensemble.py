# src/evaluate_ensemble.py
import os, argparse, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay,
    classification_report
)

from src.logger import logger
from src.loader import load_honeypot_logs, process_logs
from src.f_engineer import engineer
from src.models.ensemble import EnsembleIDS

RANDOM_STATE = 42
OUT_DIR_DEFAULT = os.path.join(os.path.dirname(__file__), "figs")

TEXT_COL = "commands" # Command text
NUMERIC_COLS = ["ip_encoded", "num_commands", "suspicious_token_hits", "avg_command_len", "duration"] # Numeric

# Load Cowrie + engineer
def load_cowrie_df(path):
    logs = load_honeypot_logs(path)
    df = process_logs(logs)
    df = engineer(df)
    return df.dropna(subset=[TEXT_COL]).copy() # Drop blanks

# Split into X/y
def split_xy(df):
    X = df[[TEXT_COL] + NUMERIC_COLS].copy() # Features
    y = df["is_malicious"].astype(int).values # Labels
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

# Train + evaluate ensemble
def fit_eval_model(X_tr, X_te, y_tr, y_te, **params):
    model = EnsembleIDS( # Ensemble wrapper
        tfidf_max_features=params.get("tfidf_max_features", 20000),
        tfidf_ngram_max=params.get("tfidf_ngram_max", 2),
        tfidf_min_df=params.get("tfidf_min_df", 2),
        rf_estimators=params.get("rf_estimators", 300),
        random_state=RANDOM_STATE,
    )
    logger.info("[*] Training Ensemble")
    model.train(X_tr, y_tr)
    logger.info("[*] Evaluating Ensemble")
    y_pred = model.predict(X_te)

    # Robust probas for ROC/PR
    try:
        y_score = model.model.predict_proba(X_te)[:, 1]
    except Exception:
        y_score = y_pred.astype(float)

    rep = classification_report(y_te, y_pred, digits=4) # Full report
    print(rep)

    metrics = dict( # Store metrics
        f1=f1_score(y_te, y_pred),
        auroc=roc_auc_score(y_te, y_score),
        auprc=average_precision_score(y_te, y_score),
    )
    return model, y_pred, y_score, metrics
