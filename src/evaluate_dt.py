import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split

from src.loader import load_honeypot_logs, process_logs
from src.f_engineer import engineer

# ------------------------------
# config
OUT_DIR = os.path.join(os.path.dirname(__file__), "figs") # Save directory
os.makedirs(OUT_DIR, exist_ok=True)

DEPTHS = [2, 4, 6, 8, 10, None] # Depth sweep
SHALLOW, DEEP = 2, 10           # Depths for confusion matrices
RANDOM_STATE = 42               # Reproducibility
# ------------------------------

def main(data_path=None):
    # load + engineer features
    if data_path:
        if data_path.endswith((".json", ".jsonl")):
            logs = load_honeypot_logs(data_path) # Load Cowrie
            df = process_logs(logs)              # Process logs
        elif data_path.endswith(".csv"):
            df = pd.read_csv(data_path)          # Load CSV
        else:
            raise ValueError("unsupported format")
    else:
        raise ValueError("provide --data path to json/jsonl/csv")

    df = engineer(df)                            # Engineer features
    X = df[["ip_encoded", "num_commands"]]       # Select features
    y = df["is_malicious"].astype(int)           # Labels

    # Train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    # ------------------------------
    # 1. validation curve
    f1s, aurocs, auprcs = [], [], []
    for d in DEPTHS:
        clf = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE) # Create DT
        clf.fit(X_tr, y_tr) # Train
        y_pred = clf.predict(X_te) # Predict
        y_score = clf.predict_proba(X_te)[:, 1] # Probabilities

        f1s.append(f1_score(y_te, y_pred))                  # F1-score
        aurocs.append(roc_auc_score(y_te, y_score))         # AUROC
        auprcs.append(average_precision_score(y_te, y_score)) # AUPRC

    # Plot validation curve
    plt.figure()
    plt.plot([str(d) for d in DEPTHS], f1s, marker="o", label="F1")
    plt.plot([str(d) for d in DEPTHS], aurocs, marker="o", label="AUROC")
    plt.plot([str(d) for d in DEPTHS], auprcs, marker="o", label="AUPRC")
    plt.xlabel("Tree Depth (max_depth)")
    plt.ylabel("Score")
    plt.title("Decision Tree Validation Curve")
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, "dt_validation_curve.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # ------------------------------
    # 2. confusion matrices
    for d in [SHALLOW, DEEP]:
        clf = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        cm = confusion_matrix(y_te, y_pred) # Confusion matrix
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(values_format="d", cmap="Blues")
        plt.title(f"Confusion Matrix — Depth = {d}")
        plt.savefig(os.path.join(OUT_DIR, f"dt_confusion_depth{d}.png"), dpi=200, bbox_inches="tight")
        plt.close()

    # ------------------------------
    # 3. feature importance
    clf = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
    clf.fit(X_tr, y_tr)
    importances = clf.feature_importances_ # Feature weights
    features = X_tr.columns

    # Plot feature importance
    plt.figure()
    plt.bar(features, importances, width=0.1)
    plt.ylabel("Importance (%)")
    plt.title("Decision Tree Feature Importances")
    plt.savefig(os.path.join(OUT_DIR, "dt_feature_importance.png"), dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[ok] saved graphs to {OUT_DIR}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to dataset (json/jsonl/csv)") # Path input
    args = ap.parse_args()
    main(args.data)
