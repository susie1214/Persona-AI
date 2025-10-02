# eval/evaluate.py
import json
from collections import Counter


def _norm(s):
    return " ".join(s.lower().split())


def f1_sets(gold_set, pred_set):
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def as_tuples_actions(items):
    # (owner, task)만 비교(기한은 누락 가능, 확장가능)
    res = set()
    for it in items:
        res.add((_norm(it.get("owner", "")), _norm(it.get("task", ""))))
    return res


def as_tuples_sentences(items):
    return set([_norm(x) for x in items])


def load_jsonl(path):
    by_id = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            by_id[o["meeting_id"]] = o
    return by_id


def evaluate(gold_path, pred_path):
    gold = load_jsonl(gold_path)
    pred = load_jsonl(pred_path)
    ks = sorted(set(gold.keys()) & set(pred.keys()))
    agg = {"actions": [], "decisions": []}
    for k in ks:
        g = gold[k]
        p = pred[k]
        # Actions
        pA, rA, fA = f1_sets(
            as_tuples_actions(g.get("actions", [])),
            as_tuples_actions(p.get("actions", [])),
        )
        # Decisions
        pD, rD, fD = f1_sets(
            as_tuples_sentences(g.get("decisions", [])),
            as_tuples_sentences(p.get("decisions", [])),
        )
        agg["actions"].append((pA, rA, fA))
        agg["decisions"].append((pD, rD, fD))

    def avg(triples):
        n = len(triples)
        return tuple(sum(x[i] for x in triples) / max(n, 1) for i in range(3))

    return {"actions": avg(agg["actions"]), "decisions": avg(agg["decisions"])}


if __name__ == "__main__":
    res = evaluate("eval/examples/gold.jsonl", "eval/examples/pred_openai.jsonl")
    print("Actions P/R/F1:", res["actions"])
    print("Decisions P/R/F1:", res["decisions"])
