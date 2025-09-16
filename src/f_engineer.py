# Import modules
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.logger import logger

# Set suspicious tokens list
SUSPICIOUS_TOKENS = {
    "wget","curl","nc","netcat", "ncat","bash","sh","python","perl",
    "chmod","chown","scp","tftp","busybox","apt","yum","ssh","telnet","pacman"
}


def _count_tokens(cmds: str) -> int: # Creates score representeing suspicious tokens
    if not cmds:
        return 0
    return sum(tok in cmds for tok in SUSPICIOUS_TOKENS)  # Returns risk indicator

def _avg_cmd_len(cmds: str) -> float: # Calculates mean character length of commands
    if not cmds:
        return 0.0
    cmds_split = [c.strip() for c in str(cmds).split(";") if c.strip()] # Splits cmds by semicolon
    if not cmds_split:
        return 0.0
    return float(np.mean([len(p) for p in cmds_split])) # Returns consistency indicator

def engineer(df: pd.DataFrame) -> pd.DataFrame: # Main feature 
    logger.info("[*] Engineering features") # Sends to serial
    df = df.copy() # Refreshes dataframe

    df["commands"] = df["commands"].fillna("")
    df["num_commands"] = df["commands"].apply(lambda x: len([p for p in str(x).split(";") if p.strip()])) # Normalise commands and counts

    df["suspicious_token_hits"] = df["commands"].apply(_count_tokens)
    df["avg_command_len"] = df["commands"].apply(_avg_cmd_len)
    df["duration"] = pd.to_numeric(df.get("duration", 0.0), errors="coerce").fillna(0.0)
    # Coverts risk and duration into temporal signal

    df["src_ip"] = df["src_ip"].fillna("0.0.0.0").astype(str)
    df["ip_encoded"] = LabelEncoder().fit_transform(df["src_ip"])
    # Converts source IP into number categorty

    if "is_malicious" in df.columns:
        df["is_malicious"] = df["is_malicious"].fillna(0).astype(int) 
        # Sets malicious flag

    return df[[
        "commands", "ip_encoded", "num_commands",
        "suspicious_token_hits", "avg_command_len", "duration",
        *(["is_malicious"] if "is_malicious" in df.columns else []) # Returns updated DF
    ]]
