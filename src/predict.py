# Import modules
import os, pandas as pd
from src.loader import load_honeypot_logs, process_logs
from src.f_engineer import engineer
from src.models.decision_tree import DTIDS
from src.models.ensemble import EnsembleIDS
from src.models.cnn import CNNIDS

# Load features from inputted model
def _load_input(src):
    if isinstance(src, pd.DataFrame):
        df = src.copy()
    elif isinstance(src, str):
        p = src.lower()
        if p.endswith(".json") or p.endswith(".jsonl"):
            logs = load_honeypot_logs(src)
            df = process_logs(logs)
        elif p.endswith(".csv"):
            df = pd.read_csv(src)
        else:
            raise ValueError("Unsupported input format")
    else:
        raise ValueError("Unsupported input source")
    return engineer(df)

# Make predictions based from features from persisted model
def predict_new_data(input_source, model_name: str):
    df = _load_input(input_source)
    if model_name == "dt":
        X = df[["ip_encoded","num_commands"]]
        m = DTIDS()
        m.load_model(os.path.abspath("saved_models/decision_tree.joblib"))
        return m.predict(X)
    if model_name == "ensemble":
        X = df[["commands","ip_encoded","num_commands","suspicious_token_hits","avg_command_len","duration"]]
        m = EnsembleIDS()
        m.load_model(os.path.abspath("saved_models/ensemble_ids.joblib"))
        return m.predict(X)
    if model_name == "cnn":
        texts = df["commands"].astype(str).tolist()
        m = CNNIDS()
        m.load(
            os.path.abspath("saved_models/cnn_ids.keras"),
            os.path.abspath("saved_models/cnn_tokenizer.joblib"),
            os.path.abspath("saved_models/cnn_meta.json"),
        )
        return m.predict(texts)
    raise ValueError("Unknown model name")
