import hashlib
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

from ai_lead_scoring.config import CHECKPOINT_DIR


def init_session_state():
    defaults = {
        "current_page": "Dashboard",
        "openai_key": "",
        "icp_product_description": "",
        "icp_target_customer": "",
        "icp_pain_points": "",
        "icp_keywords": "",
        "raw_leads": [],
        "scored_leads": [],
        "scoring_errors": [],
        "upload_columns": [],
        "col_title": "",
        "col_body": "",
        "col_permalink": "",
        "col_author": "",
        "col_engagement": "",
        "activity_log": [],
        "jsonl_paths": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def add_log(message: str, status: str = "Completed"):
    st.session_state.activity_log.insert(0, {"timestamp": now_iso(), "activity": message, "status": status})
    st.session_state.activity_log = st.session_state.activity_log[:200]


def save_jsonl(path: str, records: List[Dict[str, Any]]) -> str:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def dataset_fingerprint(leads: List[Dict[str, Any]]) -> str:
    sample = json.dumps(leads[:5], sort_keys=True, ensure_ascii=False)
    return hashlib.md5(f"{len(leads)}|{sample}".encode()).hexdigest()[:8]


def checkpoint_path(fp: str) -> str:
    return os.path.join(CHECKPOINT_DIR, f"checkpoint_{fp}.jsonl")


def checkpoint_exists(fp: str) -> bool:
    return os.path.isfile(checkpoint_path(fp))


def load_checkpoint(fp: str) -> Dict[str, Dict]:
    path = checkpoint_path(fp)
    score_map: Dict[str, Dict] = {}
    if not os.path.isfile(path):
        return score_map
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    score_map[str(rec["id"])] = rec
                except Exception:
                    continue
    return score_map


def append_checkpoint(fp: str, results: List[Dict[str, Any]]):
    with open(checkpoint_path(fp), "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def clear_checkpoint(fp: str):
    path = checkpoint_path(fp)
    if os.path.isfile(path):
        os.remove(path)


def list_all_checkpoints() -> List[Dict[str, Any]]:
    found = []
    for fname in os.listdir(CHECKPOINT_DIR):
        if fname.startswith("checkpoint_") and fname.endswith(".jsonl"):
            fpath = os.path.join(CHECKPOINT_DIR, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    lines = [l for l in f if l.strip()]
                fp = fname.replace("checkpoint_", "").replace(".jsonl", "")
                mtime = datetime.utcfromtimestamp(os.path.getmtime(fpath)).strftime("%Y-%m-%d %H:%M UTC")
                found.append({"fingerprint": fp, "file": fname, "scored_so_far": len(lines), "last_modified": mtime})
            except Exception:
                continue
    return found


def _extract_json_array(text: str) -> List[Dict[str, Any]]:
    t = re.sub(r"^```[a-zA-Z]*\s*", "", text.strip())
    t = re.sub(r"\s*```$", "", t).strip()
    start, end = t.find("["), t.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array found: {t[:300]}")
    return json.loads(t[start:end + 1])


def parse_uploaded_file(uploaded_file) -> Tuple[List[Dict[str, Any]], List[str]]:
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            return df.to_dict(orient="records"), list(df.columns)
        elif name.endswith(".jsonl"):
            content = uploaded_file.read().decode("utf-8")
            records = [json.loads(line) for line in content.splitlines() if line.strip()]
            return records, list(records[0].keys()) if records else []
        elif name.endswith(".json"):
            data = json.loads(uploaded_file.read().decode("utf-8"))
            records = data if isinstance(data, list) else next((v for v in data.values() if isinstance(v, list)), [data])
            return records, list(records[0].keys()) if records else []
        return [], []
    except Exception as e:
        st.error(f"Failed to parse file: {e}")
        return [], []
