# Import modules
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from src.loader import load_honeypot_logs, process_logs
from src.f_engineer import engineer

# Define UNSW Features
UNSW_CATEGORICAL = ["proto", "service", "state"]
UNSW_NUMERIC = [
    "dur","spkts","dpkts","sbytes","dbytes","rate","sttl","dttl","sload","dload",
    "sloss","dloss","sinpkt","dinpkt","sjit","djit","swin","stcpb","dtcpb","dwin",
    "tcprtt","synack","ackdat","smeansz","dmeansz","trans_depth","res_bdy_len",
    "ct_srv_src","ct_state_ttl","ct_dst_ltm","ct_src_dport_ltm","ct_dst_sport_ltm","ct_dst_src_ltm"
]

# Detect UNNW or Cowrie dataset
def _unsw_detect(df: pd.DataFrame) -> bool:
    cols = set(c.strip().lower() for c in df.columns)
    return "label" in cols and ({"proto","service","state"}.issubset(cols) or "attack_cat" in cols)

# Extract numeric categories
def _bin_numeric(col: pd.Series, nbins: int = 8) -> pd.Series:
    col = pd.to_numeric(col, errors="coerce").fillna(0)
    try:
        q = pd.qcut(col, q=nbins, labels=False, duplicates="drop")
    except Exception:
        q = pd.cut(col, bins=nbins, labels=False, include_lowest=True)
    return q.fillna(0).astype(int)

# Extrat sequential categories
def _row_to_text_unsw(row: pd.Series) -> str:
    toks = []
    for c in UNSW_CATEGORICAL:
        if c in row.index:
            val = str(row.get(c, "")).strip().lower()
            if val and val != "nan":
                toks.append(f"{c}={val}")

    for cat in UNSW_NUMERIC:
        if cat in row.index:
            val = row.get(cat)
            if pd.notna(val):
                toks.append(f"{cat}_bin={int(val)}")
    return " ".join(toks)

# Engineer numeric and sequential features
def _prepare_unsw_texts(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "label" not in df.columns:
        raise ValueError("UNSW: expected 'label' column with 0/1 labels.")
    y = df["label"].astype(int).values

    for c in UNSW_NUMERIC:
        if c in df.columns:
            df[c] = _bin_numeric(df[c])

    if "attack_cat" not in df.columns:
        df["attack_cat"] = np.where(df["label"].astype(int) == 1, "attack", "normal")

    texts = [_row_to_text_unsw(row) for _, row in df.iterrows()]
    return texts, y


# Determine benignity
def load_text_dataset(data_path: str | None) -> tuple[list[str], np.ndarray]:
    if data_path and data_path.lower().endswith((".jsonl", ".json")):
        logs = load_honeypot_logs(data_path)
        df = process_logs(logs)
        df = engineer(df)
        if "commands" not in df.columns or "is_malicious" not in df.columns:
            raise ValueError("Cowrie pipeline expects 'commands' and 'is_malicious'.")
        texts = df["commands"].astype(str).tolist()
        y = df["is_malicious"].astype(int).values
        return texts, y

    if data_path and data_path.lower().endswith(".csv"):
        df = pd.read_csv(data_path, low_memory=False)
        if _unsw_detect(df):
            return _prepare_unsw_texts(df)
        cols_lower = [c.lower() for c in df.columns]
        if "text" in cols_lower and "label" in cols_lower:
            tcol = df.columns[cols_lower.index("text")]
            lcol = df.columns[cols_lower.index("label")]
            texts = df[tcol].astype(str).tolist()
            y = df[lcol].astype(int).values
            return texts, y
        raise ValueError("CSV not recognised as UNSW or generic 'text'/'label' format.")

    # LOAD COWRIE LOGS IF NO PATH SPECIFID
    logs = load_honeypot_logs()
    df = process_logs(logs)
    df = engineer(df)
    if "commands" not in df.columns or "is_malicious" not in df.columns:
        raise ValueError("Cowrie pipeline expects 'commands' and 'is_malicious'.")
    texts = df["commands"].astype(str).tolist()
    y = df["is_malicious"].astype(int).values
    return texts, y
