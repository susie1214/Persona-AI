# -*- coding: utf-8 -*-
"""
style_eval.py
-------------
두 문장(ref_text, pred_text) 간 페르소나 스타일 유사도를 자동 채점하는 스크립트.

입력:
  --ref   : 정답 스타일 정의 CSV
            (id, persona_id, question, ref_text, rules, signature_tokens)
  --pred  : 특정 구간(5분/15분/30분/60분 등)의 실제 답변 CSV
            (id, persona_id, question, ref_text)
            * 여기 ref_text = 실제 답변 (pred_text 역할)

출력:
  --output: 평가 결과 CSV
    컬럼:
      id, persona_id, question,
      ref_text, pred_text,
      rules, signature_tokens,
      style_sim, rule_score, sig_token_score, final_score

의존성:
  - python 3.x
  - (선택) pip install sentence-transformers
    -> 없으면 style_sim은 ""로 두고 나머지 지표만 계산.

시간 복잡도:
  - N개의 샘플, 각 문장 길이를 L이라 할 때
  - 전처리 및 스코어 계산: O(N * L)
  - 임베딩 계산: O(N * L) (모델 서열 길이에 비례)
"""

import argparse
import csv
import math
from typing import Dict, Tuple, List, Optional

# sentence-transformers는 선택
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _ST_AVAILABLE = True
except Exception:
    SentenceTransformer = None
    np = None
    _ST_AVAILABLE = False


# -----------------------------
# 유틸: CSV 로딩 / 조인
# -----------------------------

def load_style_ref(path: str) -> Dict[Tuple[str, str], dict]:
    """
    정답 스타일 정의 파일 로드.
    key: (persona_id, question)
    value: row dict
    """
    mapping: Dict[Tuple[str, str], dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = (row.get("persona_id", "") or "").strip()
            q = (row.get("question", "") or "").strip()
            if not pid or not q:
                continue
            mapping[(pid, q)] = row
    return mapping


def join_ref_and_pred(
    ref_path: str,
    pred_path: str,
) -> List[dict]:
    """
    ref_csv와 pred_csv를 (persona_id, question) 기준으로 조인.

    ref_csv: id, persona_id, question, ref_text, rules, signature_tokens
    pred_csv: id, persona_id, question, ref_text(=pred_text)

    반환: 공통 key가 있는 row 리스트
    """
    ref_map = load_style_ref(ref_path)
    results: List[dict] = []

    with open(pred_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for prow in reader:
            persona_id = (prow.get("persona_id", "") or "").strip()
            question = (prow.get("question", "") or "").strip()
            pred_text = prow.get("ref_text", "") or ""

            key = (persona_id, question)
            ref_row = ref_map.get(key)

            if ref_row is None:
                # 필요하면 id 기반으로 fallback 할 수도 있음
                print(f"[WARN] No style_ref match for persona_id={persona_id}, question={question[:30]}...")
                continue

            out = {
                "id": prow.get("id", ref_row.get("id", "")),
                "persona_id": persona_id,
                "question": question,
                # 정답 스타일 텍스트
                "ref_text": ref_row.get("ref_text", ""),
                # 실제 답변
                "pred_text": pred_text,
                "rules": ref_row.get("rules", ""),
                "signature_tokens": ref_row.get("signature_tokens", ""),
            }
            results.append(out)

    print(f"[INFO] Joined {len(results)} rows (ref={ref_path}, pred={pred_path})")
    return results


# -----------------------------
# 스코어 계산 로직
# -----------------------------

def cosine_sim(a: "np.ndarray", b: "np.ndarray") -> float:
    """코사인 유사도"""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class StyleScorer:
    """
    스타일 유사도 계산기.
    - sentence-transformers가 있으면 의미 유사도(style_sim) 사용
    - 없으면 style_sim은 "" (빈 값)으로 두고 나머지만 사용
    """

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model: Optional[SentenceTransformer] = None
        if _ST_AVAILABLE:
            try:
                print(f"[INFO] Loading sentence-transformers model: {model_name}")
                self.model = SentenceTransformer(model_name)
                print("[INFO] Model loaded.")
            except Exception as e:
                print(f"[WARN] Failed to load model: {e}. style_sim will be empty.")
        else:
            print("[INFO] sentence-transformers not available. style_sim will be empty.")

    def embed_pair(self, ref_text: str, pred_text: str) -> Optional[float]:
        """두 문장의 임베딩 코사인 유사도. 실패 시 None."""
        if self.model is None:
            return None
        try:
            embs = self.model.encode([ref_text, pred_text])
            return cosine_sim(embs[0], embs[1])
        except Exception as e:
            print(f"[WARN] embedding failed: {e}")
            return None

    @staticmethod
    def _split_tokens(s: str) -> List[str]:
        """
        "a;b;c" 형태 문자열을 ["a", "b", "c"]로 분리.
        공백 제거, 빈 문자열 제거.
        """
        if not s:
            return []
        return [t.strip() for t in s.split(";") if t.strip()]

    @staticmethod
    def _ratio_match_tokens(tokens: List[str], text: str) -> float:
        """
        tokens 중 몇 개가 text에 substring으로 등장하는지 비율.
        """
        if not tokens:
            return math.nan  # 토큰이 없으면 점수 정의 불가
        text_lower = text.lower()
        hit = 0
        for tok in tokens:
            if tok.lower() in text_lower:
                hit += 1
        return hit / len(tokens)

    def score_row(self, row: dict) -> dict:
        """
        한 샘플에 대해 style_sim, rule_score, sig_token_score, final_score 계산.
        """
        ref_text = row.get("ref_text", "") or ""
        pred_text = row.get("pred_text", "") or ""
        rules = row.get("rules", "") or ""
        sigs = row.get("signature_tokens", "") or ""

        # 1) 스타일 임베딩 유사도
        sim = self.embed_pair(ref_text, pred_text)
        if sim is None:
            style_sim_str = ""  # 없으면 공백
        else:
            # 0~1 사이 값으로 가정
            style_sim_str = f"{sim:.4f}"

        # 2) 규칙 매칭 점수 (단순 substring 기준)
        rule_tokens = self._split_tokens(rules)
        rule_score = self._ratio_match_tokens(rule_tokens, pred_text) if rule_tokens else math.nan

        # 3) 시그니처 토큰 매칭 점수
        sig_tokens = self._split_tokens(sigs)
        sig_score = self._ratio_match_tokens(sig_tokens, pred_text) if sig_tokens else math.nan

        # 4) 최종 점수: 사용 가능한 지표들의 평균
        scores_for_avg: List[float] = []
        if sim is not None:
            scores_for_avg.append(sim)
        if not math.isnan(rule_score):
            scores_for_avg.append(rule_score)
        if not math.isnan(sig_score):
            scores_for_avg.append(sig_score)

        if scores_for_avg:
            final_score = sum(scores_for_avg) / len(scores_for_avg)
        else:
            final_score = math.nan

        # 문자열로 변환 (CSV에 넣기 위해)
        row["style_sim"] = style_sim_str
        row["rule_score"] = "" if math.isnan(rule_score) else f"{rule_score:.4f}"
        row["sig_token_score"] = "" if math.isnan(sig_score) else f"{sig_score:.4f}"
        row["final_score"] = "" if math.isnan(final_score) else f"{final_score:.4f}"

        return row


# -----------------------------
# 메인
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Persona style evaluation (ref CSV + pred CSV)")
    parser.add_argument("--ref", required=True, help="정답 스타일 정의 CSV 경로 (예: style_ref.csv)")
    parser.add_argument("--pred", required=True, help="예측(실제 답변) CSV 경로 (예: logs_15min.csv)")
    parser.add_argument("--output", required=True, help="평가 결과 CSV 경로")
    parser.add_argument("--model_name", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                        help="sentence-transformers 모델 이름")

    args = parser.parse_args()

    # 1) ref + pred 조인
    joined = join_ref_and_pred(args.ref, args.pred)

    # 2) 스코어링
    scorer = StyleScorer(model_name=args.model_name)
    scored_rows = [scorer.score_row(row) for row in joined]

    # 3) CSV 저장
    fieldnames = [
        "id",
        "persona_id",
        "question",
        "ref_text",
        "pred_text",
        "rules",
        "signature_tokens",
        "style_sim",
        "rule_score",
        "sig_token_score",
        "final_score",
    ]

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scored_rows)

    print(f"[INFO] Wrote {len(scored_rows)} scored rows -> {args.output}")


if __name__ == "__main__":
    main()
