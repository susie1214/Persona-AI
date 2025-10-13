# core/charts.py
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

def speaker_talktime_bar(utterances, png_path):
    dur = defaultdict(float)
    for u in utterances:
        if "start" in u and "end" in u:
            dur[u["speaker"]] += max(0.0, float(u["end"])-float(u["start"]))
    labels = list(dur.keys()); vals = [dur[k] for k in labels]
    plt.figure()
    plt.bar(labels, vals)
    plt.title("화자별 발화 시간(초)")
    plt.xlabel("Speaker"); plt.ylabel("Duration (s)")
    plt.tight_layout()
    plt.savefig(png_path); plt.close()

def owner_todo_bar(action_items, png_path):
    from collections import Counter
    owners = [ai.owner or "미지정" for ai in action_items]
    c = Counter(owners)
    labels, vals = zip(*c.items()) if c else ([], [])
    plt.figure()
    plt.bar(labels, vals)
    plt.title("담당자별 Action Items 수")
    plt.xlabel("Owner"); plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(png_path); plt.close()
