# -*- coding: utf-8 -*-
"""
plot_style_curve.py
-------------------
여러 개의 style_eval 출력 CSV에서 final_score 평균을 구해
회의 길이(분)별 페르소나 적응 곡선을 그리는 스크립트.

입력:
  --conds 5,10,30,60
  --files style_eval_5min.csv,style_eval_10min.csv,style_eval_30min.csv,style_eval_60min.csv

출력:
  style_curve.png  (현재 디렉터리에 저장)

시간복잡도: O(N) (N = 모든 CSV 행 수의 합)
공간복잡도: O(K) (K = 조건 개수, 여기서는 4)
"""

import csv
import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


def parse_list(arg: str) -> List[str]:
    """쉼표로 구분된 문자열을 리스트로 변환."""
    if not arg:
        return []
    parts = [x.strip() for x in arg.split(",")]
    return [x for x in parts if x]


def read_final_scores(csv_path: Path) -> List[float]:
    """CSV에서 final_score 컬럼을 읽어 리스트로 반환."""
    scores: List[float] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get("final_score", "")
            if not val:
                continue
            try:
                scores.append(float(val))
            except ValueError:
                continue
    return scores


def calc_mean(values: List[float]) -> float:
    """리스트 평균 계산. 값이 없으면 0.0."""
    if not values:
        return 0.0
    s = 0.0
    for v in values:
        s += v
    return s / len(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conds",
        type=str,
        required=True,
        help="회의 길이(분) 리스트. 예: 5,10,30,60",
    )
    parser.add_argument(
        "--files",
        type=str,
        required=True,
        help="각 길이에 대응하는 CSV 경로 리스트. 예: style_eval_5min.csv,style_eval_10min.csv,...",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="style_curve.png",
        help="그래프 저장 파일명 (기본: style_curve.png)",
    )
    args = parser.parse_args()

    cond_strs = parse_list(args.conds)
    file_strs = parse_list(args.files)

    if len(cond_strs) != len(file_strs):
        print("[ERROR] conds 개수와 files 개수가 다릅니다.")
        print("  conds:", cond_strs)
        print("  files:", file_strs)
        return

    # (길이, 평균점수) 쌍 계산
    points: List[Tuple[int, float]] = []

    for c_str, f_str in zip(cond_strs, file_strs):
        try:
            minutes = int(c_str)
        except ValueError:
            print(f"[WARN] 정수가 아닌 cond 무시: {c_str}")
            continue

        csv_path = Path(f_str)
        if not csv_path.exists():
            print(f"[WARN] 파일이 존재하지 않아 건너뜁니다: {csv_path}")
            continue

        scores = read_final_scores(csv_path)
        mean_score = calc_mean(scores)

        print(f"[INFO] {minutes}분: 샘플수={len(scores)}, 평균 final_score={mean_score:.4f}")
        points.append((minutes, mean_score))

    if not points:
        print("[ERROR] 유효한 데이터가 없습니다.")
        return

    # 길이 순으로 정렬
    points.sort(key=lambda x: x[0])
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    # 그래프 그리기
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("회의 길이 (분)")
    plt.ylabel("평균 Persona Style Score (final_score)")
    plt.title("회의 길이에 따른 페르소나 적응 곡선")
    plt.grid(True)

    out_path = Path(args.out)
    plt.savefig(out_path, bbox_inches="tight")
    print(f"[OK] 그래프 저장 완료: {out_path}")


if __name__ == "__main__":
    main()
