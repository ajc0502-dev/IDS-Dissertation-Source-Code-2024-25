# Import Modules
import json
from collections import defaultdict
import pandas as pd
from src.logger import logger
from src.config import DATA_PATH

#Set suspicious tokens list
SUSPICIOUS_TOKENS = {
    "wget","curl","nc","netcat", "ncat","bash","sh","python","perl",
    "chmod","chown","scp","tftp","busybox","apt","yum","ssh","telnet","pacman"
}

# load_honeypot_logs() - Proccess the Cowrie logs into a JSON format
def load_honeypot_logs(path: str = DATA_PATH):
    logger.info(f"[*] Loading logs from {path}")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line)) # Load JSON lines and append into JSONL
            except json.JSONDecodeError:
                logger.warning(f"[-] Error! Invalid JSON line. Skipping {ln}")
    return records

#process_logs() - Load logs and extract important features
def process_logs(logs):
    logger.info("[*] Processing logs into features")
    sessions = defaultdict(lambda: {
        "src_ip": None, # Source IP
        "username": None, # Username
        "password": None, # Password
        "commands": [], # Predefined CMD List
        "duration": 0.0, # Session duration
        "login_success": False, # Successful login
        "login_failed": False,# Failed login
    })

    # Reads the log and sepeates into feature lists by session ID
    for ev in logs:
        session_id = ev.get("session")
        if not session_id:
            continue
        event = ev.get("eventid","")
        # Extracts the SRC IP upon session connection
        if event == "cowrie.session.connect":
            sessions[session_id]["src_ip"] = ev.get("src_ip") or sessions[session_id]["src_ip"]

        # Extracts the session authentication success and logs pasword attempts
        elif event in ("cowrie.login.failed","cowrie.login.success"):
            sessions[session_id]["username"] = ev.get("username") or sessions[session_id]["username"]
            sessions[session_id]["password"] = ev.get("password") or sessions[session_id]["password"]
            if event == "cowrie.login.success":
                sessions[session_id]["login_success"] = True
            else:
                sessions[session_id]["login_failed"] = True
        # Extracts the inputted commands 
        elif event == "cowrie.command.input":
            cmd = ev.get("input") or ev.get("command")
            if cmd:
                sessions[session_id]["commands"].append(str(cmd))
        # Extracts the session termination
        elif event == "cowrie.session.closed":
            try:
                sessions[session_id]["duration"] = float(ev.get("duration", 0.0))
            except Exception:
                sessions[session_id]["duration"] = 0.0

    rows = []
    for session_id, s in sessions.items():
        cmds = " ; ".join(s["commands"]) # Seperates CMDSs by semi-colon
        has_suspicious = any(tok in cmds for tok in SUSPICIOUS_TOKENS)
        is_malicious = int(s["login_failed"] or has_suspicious) # Appends malicious
        rows.append({ # Appends extracted feautes and malicious flags into a new list
            "session": session_id,
            "src_ip": s["src_ip"],
            "username": s["username"],
            "password": s["password"],
            "commands": cmds,
            "duration": s["duration"],
            "is_malicious": is_malicious
        })

    df = pd.DataFrame(rows) # Formats the list into a DataFrame
    logger.info(f"[*] Built {len(df)} session entries") # Logs to console
    return df
