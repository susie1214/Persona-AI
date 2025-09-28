# core/persona.py
import os, json


def load_persona(user_id="jkj"):
    p = f"data/persona/{user_id}.json"
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "style": {
            "tone": "정중/공식",
            "format": "개조식",
            "sentence_len": "적당히",
            "jargon": "",
        },
        "alerts": {"remind": "30분 전", "fields": ["제목", "참석자"]},
        "consent": False,
    }


def save_persona(user_id, data):
    os.makedirs("data/persona", exist_ok=True)
    with open(f"data/persona/{user_id}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
