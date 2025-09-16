# Import Modules
import os, argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
)
from src.loader import load_honeypot_logs, process_logs
from src.f_engineer import engineer
from src.models.cnn import CNNIDS

# Ensure output directory
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

# Load dataset + engineer
def load_dataset(data_path: str):
    if data_path and data_path.lower().endswith((".jsonl", ".json")):
        logs = load_honeypot_logs(data_path) # Cowrie
        df = process_logs(logs)
    elif data_path and data_path.lower().endswith(".csv"):
        df = pd.read_csv(data_path) # UNSW
    else:
        logs = load_honeypot_logs() # Default fallback
        df = process_logs(logs)

    df = engineer(df) # Engineer features
    if "commands" not in df.columns: raise ValueError("no commands col")
    if "is_malicious" not in df.columns: raise ValueError("no label col")

    texts = df["commands"].astype(str).tolist() # Commands
    y = df["is_malicious"].astype(int).values   # Labels
    return texts, y

# PR + ROC plot
def plot_roc_pr(y_true, y_score, out_dir, tag):
    fpr, tpr, _ = roc_curve(y_true, y_score) # ROC curve
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUROC = {roc_auc:.3f}") # Plot ROC
    plt.plot([0, 1], [0, 1], linestyle="--") # Baseline
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"CNN ROC — {tag}"); plt.legend(loc="lower right")
    plt.savefig(os.path.join(out_dir, f"cnn_roc_{tag}.png"), dpi=200, bbox_inches="tight")
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_score) # PR curve
    ap = average_precision_score(y_true, y_score)
    plt.figure()
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"CNN Precision–Recall — {tag}"); plt.legend(loc="lower left")
    plt.savefig(os.path.join(out_dir, f"cnn_pr_{tag}.png"), dpi=200, bbox_inches="tight")
    plt.close()
    return roc_auc, ap

# Confusion matrix plot
def plot_confusion(y_true, y_pred, out_dir, tag):
    cm = confusion_matrix(y_true, y_pred) # Create CM
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(values_format="d", cmap="Blues")
    plt.title(f"CNN Confusion Matrix — {tag}")
    plt.savefig(os.path.join(out_dir, f"cnn_confusion_{tag}.png"), dpi=200, bbox_inches="tight")
    plt.close()

# Threshold selection by macro-F1
def choose_threshold(y_true, probs):
    ts = np.linspace(0.05, 0.95, 19) # Range
    scores = [f1_score(y_true, (probs >= t).astype(int), average="macro", zero_division=0) for t in ts]
    return float(ts[int(np.argmax(scores))]) # Best threshold

# Run one config
OUT_DIR_DEFAULT = os.path.join(os.path.dirname(__file__), "figs")
def run_once(texts, y, max_len, vocab_size, epochs, batch_size,
             random_state=42, limit=None, out_dir=OUT_DIR_DEFAULT):
    # Optional subsample
    if limit and limit < len(texts):
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(texts), size=limit, replace=False)
        texts, y = [texts[i] for i in idx], y[idx]

    # Train/val/test split
    X_tr, X_te, y_tr, y_te = train_test_split(texts, y, test_size=0.25, stratify=y, random_state=random_state)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=random_state)

    cnn = CNNIDS(max_len=max_len, vocab_size=vocab_size) # Build CNN
    history = cnn.fit(X_tr, y_tr, X_va, y_va, epochs=epochs, batch_size=batch_size) # Train

    # Threshold tuning
    va_prob = cnn.model.predict(cnn._texts_to_sequences(X_va), verbose=0).ravel()
    thr = choose_threshold(y_va, va_prob) # Pick threshold
    print(f"[eval] chosen threshold from validation = {thr:.3f}")

    # Test evaluation
    y_score = cnn.model.predict(cnn._texts_to_sequences(X_te), verbose=0).ravel()
    y_pred = (y_score >= thr).astype(int)

    print(classification_report(y_te, y_pred, digits=4, zero_division=0)) # Report
    tag = f"len{max_len}_v{vocab_size}"
    roc_auc, ap = plot_roc_pr(y_te, y_score, out_dir, tag) # Save ROC/PR
    plot_confusion(y_te, y_pred, out_dir, tag)             # Save CM

    # Return results for heatmap
    report = classification_report(y_te, y_pred, digits=4, output_dict=True, zero_division=0)
    return {"max_len": max_len, "vocab_size": vocab_size, "accuracy": report["accuracy"],
            "f1_macro": report["macro avg"]["f1-score"], "auroc": roc_auc, "auprc": ap}
