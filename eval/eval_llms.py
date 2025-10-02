# -*- coding: utf-8 -*-
"""
eval/eval_llms.py
- 프로브(prompts) JSONL을 읽어 3개 모델(OpenAI / skt A.X / Midm)을 실행/캐시
- 모델 출력 → 라벨 매핑(키워드 기반) → Precision/Recall/F1 계산(마이크로/매크로)
- 결과를 콘솔 표 + CSV로 저장

입력:
  data/prompts.jsonl           # [{"id": "q1", "prompt": "...", "gold": ["LABEL_A","LABEL_B"]}, ...]
  eval/labels.yaml             # 라벨→키워드 매핑 (모델 출력 텍스트를 라벨로 변환)
출력:
  predictions/{model}.jsonl    # 모델별 예측 캐시
  eval_reports/report_<ts>.csv  # 최종 리포트

사용 예:
  (venv) python -m eval.eval_llms --mode all
  (venv) python -m eval.eval_llms --mode predict
  (venv) python -m eval.eval_llms --mode eval
"""
import os, json, argparse, datetime, csv, sys, re
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple

try:
    import yaml
except Exception as e:
    print("[INFO] PyYAML 미설치 → 설치 필요: pip install pyyaml"); raise

# 본 프로젝트 래퍼
try:
    from core.llm_openai import OpenAILLM
    from core.llm_ax import AXLLM
    from core.llm_midm import MidmLLM
except Exception as e:
    print("[ERROR] core.* LLM 래퍼 import 실패. 프로젝트 구조/경로를 확인하세요:", e)
    raise

# ----------------------------- IO -----------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EVAL_DIR = ROOT / "eval"
PROMPTS_PATH = DATA_DIR / "prompts.jsonl"
LABELS_YAML = EVAL_DIR / "labels.yaml"
PRED_DIR = ROOT / "predictions"
REPORT_DIR = ROOT / "eval_reports"
PRED_DIR.mkdir(exist_ok=True, parents=True)
REPORT_DIR.mkdir(exist_ok=True, parents=True)

MODELS = {
    "OpenAI": lambda: OpenAILLM("gpt-4o-mini"),
    "A.X":    lambda: AXLLM("skt/A.X-4.0"),
    "Midm":   lambda: MidmLLM("K-intelligence/Midm-2.0-Mini-Instruct"),
}

# ---------------------- helpers: file read/write ----------------------
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            items.append(json.loads(line))
    return items

def write_jsonl(path: Path, items: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def load_labels_yaml(path: Path) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # to lowercase keyword list
    return {lbl: [kw.lower() for kw in kws] for lbl, kws in data.items()}

# ---------------------- labeling: text -> labels ----------------------
def text_to_labels(text: str, label_kws: Dict[str, List[str]]) -> Set[str]:
    """모델 출력 텍스트를 라벨 세트로 변환 (키워드 OR 매칭, 대소문자 무시)."""
    t = (text or "").lower()
    found = set()
    for lbl, kws in label_kws.items():
        if any(kw in t for kw in kws):
            found.add(lbl)
    return found

# ---------------------- metrics ----------------------
def f1_micro_macro(all_gold: List[Set[str]], all_pred: List[Set[str]]) -> Dict[str, float]:
    """멀티라벨 마이크로/매크로 P/R/F1."""
    labels = sorted(list({l for s in all_gold for l in s} | {l for s in all_pred for l in s}))
    # per-label stats
    per = {}
    for L in labels:
        tp = sum(1 for g, p in zip(all_gold, all_pred) if (L in g and L in p))
        fp = sum(1 for g, p in zip(all_gold, all_pred) if (L not in g and L in p))
        fn = sum(1 for g, p in zip(all_gold, all_pred) if (L in g and L not in p))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = (2*prec*rec)/(prec+rec) if (prec+rec) else 0.0
        per[L] = {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}

    # micro
    TP = sum(v["tp"] for v in per.values())
    FP = sum(v["fp"] for v in per.values())
    FN = sum(v["fn"] for v in per.values())
    micro_p = TP / (TP + FP) if (TP + FP) else 0.0
    micro_r = TP / (TP + FN) if (TP + FN) else 0.0
    micro_f1 = (2*micro_p*micro_r)/(micro_p+micro_r) if (micro_p+micro_r) else 0.0

    # macro
    macro_p = sum(v["precision"] for v in per.values())/len(per) if per else 0.0
    macro_r = sum(v["recall"] for v in per.values())/len(per) if per else 0.0
    macro_f1 = sum(v["f1"] for v in per.values())/len(per) if per else 0.0

    return {
        "micro_precision": micro_p, "micro_recall": micro_r, "micro_f1": micro_f1,
        "macro_precision": macro_p, "macro_recall": macro_r, "macro_f1": macro_f1,
        "per_label": per,
        "labels": labels,
    }

# ---------------------- prediction ----------------------
def predict_model(model_name: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """각 모델을 실행하고 결과 캐시를 반환 [{id, prompt, gold, pred_text}]"""
    print(f"\n[RUN] {model_name} 예측 생성 중...")
    runner = MODELS[model_name]()
    outs = []
    for it in items:
        pid = it["id"]
        prompt = it["prompt"]
        try:
            text = runner.complete(prompt, temperature=0.2)
        except Exception as e:
            text = f"[ERROR] {e}"
        outs.append({"id": pid, "prompt": prompt, "gold": it.get("gold", []), "pred_text": text})
    return outs

def ensure_predictions(mode: str) -> Dict[str, List[Dict[str, Any]]]:
    """mode에 따라 (predict/eval/all) predictions/{model}.jsonl 생성 또는 로드."""
    items = read_jsonl(PROMPTS_PATH)
    preds = {}
    for m in MODELS.keys():
        f = PRED_DIR / f"{m}.jsonl"
        if mode in ("all", "predict") or not f.exists():
            outs = predict_model(m, items)
            write_jsonl(f, outs)
        preds[m] = read_jsonl(f)
    return preds

# ---------------------- evaluation ----------------------
def evaluate(preds: Dict[str, List[Dict[str, Any]]], label_kws: Dict[str, List[str]]):
    results = []
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_csv = REPORT_DIR / f"report_{ts}.csv"

    print("\n===================== EVAL REPORT =====================")
    for model_name, rows in preds.items():
        gold_sets = [set(r.get("gold", [])) for r in rows]
        pred_sets = [text_to_labels(r.get("pred_text", ""), label_kws) for r in rows]

        met = f1_micro_macro(gold_sets, pred_sets)
        micro = (met["micro_precision"], met["micro_recall"], met["micro_f1"])
        macro = (met["macro_precision"], met["macro_recall"], met["macro_f1"])

        # 콘솔 출력
        print(f"\n[MODEL] {model_name}")
        print(f"  Micro  P/R/F1: {micro[0]:.4f} / {micro[1]:.4f} / {micro[2]:.4f}")
        print(f"  Macro  P/R/F1: {macro[0]:.4f} / {macro[1]:.4f} / {macro[2]:.4f}")
        print("  Per-label:")
        for L in met["labels"]:
            s = met["per_label"][L]
            print(f"    - {L:>12s} | P {s['precision']:.3f}  R {s['recall']:.3f}  F1 {s['f1']:.3f}  (TP{ s['tp']}, FP{ s['fp']}, FN{ s['fn']})")

        # CSV 축적
        results.append({
            "model": model_name,
            "micro_precision": f"{micro[0]:.6f}",
            "micro_recall":    f"{micro[1]:.6f}",
            "micro_f1":        f"{micro[2]:.6f}",
            "macro_precision": f"{macro[0]:.6f}",
            "macro_recall":    f"{macro[1]:.6f}",
            "macro_f1":        f"{macro[2]:.6f}",
        })

    # CSV 저장
    with open(report_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results: w.writerow(r)

    print("\n📄 CSV 저장:", report_csv)
    print("=======================================================\n")

# ---------------------- main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=["all","predict","eval"],
                    help="all: 예측+평가 / predict: 예측만 / eval: 캐시로 평가만")
    args = ap.parse_args()

    if not PROMPTS_PATH.exists():
        print(f"[ERROR] {PROMPTS_PATH} 가 없습니다. data/prompts.jsonl 추가해주세요.")
        sys.exit(1)
    if not LABELS_YAML.exists():
        print(f"[ERROR] {LABELS_YAML} 가 없습니다. eval/labels.yaml 추가해주세요.")
        sys.exit(1)

    label_kws = load_labels_yaml(LABELS_YAML)

    if args.mode in ("all", "predict"):
        preds = ensure_predictions(mode="all" if args.mode=="all" else "predict")
    else:
        # eval only: 기존 캐시 필요
        preds = {m: read_jsonl(PRED_DIR / f"{m}.jsonl") for m in MODELS.keys()}
        for m, rows in preds.items():
            if not rows:
                print(f"[ERROR] predictions/{m}.jsonl 이 비었습니다. 먼저 --mode predict 실행하세요.")
                sys.exit(1)

    evaluate(preds, label_kws)

if __name__ == "__main__":
    main()
