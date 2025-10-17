# core/schedule_store.py
from __future__ import annotations
import json, os, tempfile, time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional

# 저장 위치(원하면 프로젝트 루트 기준 경로를 바꿔도 됨)
STORE_PATH = os.path.join(os.getcwd(), "schedules.json")

@dataclass
class Schedule:
    id: int
    title: str
    location: Optional[str]
    meeting_start: str  # "YYYY-MM-DDTHH:MM:SS"
    meeting_end:   str  # "YYYY-MM-DDTHH:MM:SS"
    project_start: Optional[str] = None  # "YYYY-MM-DD"
    project_due:   Optional[str] = None
    settlement_at: Optional[str] = None
    todos: List[str] = None

def _read_store() -> Dict[str, Any]:
    if not os.path.exists(STORE_PATH):
        return {"version": 1, "items": []}
    with open(STORE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _atomic_write(obj: Dict[str, Any]):
    d = os.path.dirname(STORE_PATH) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix="schedules_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, STORE_PATH)
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except: pass

def new_id() -> int:
    return int(time.time() * 1000)

def save_schedule(s: Schedule) -> Schedule:
    """업서트: meeting_start + title 동일하면 갱신, 없으면 추가"""
    store = _read_store()
    items = store.get("items", [])
    idx = next(
        (i for i, it in enumerate(items)
         if it.get("meeting_start") == s.meeting_start and it.get("title") == s.title),
        -1
    )
    if idx >= 0:
        # update (id는 유지)
        s.id = items[idx].get("id", s.id)
        items[idx].update(asdict(s))
    else:
        items.append(asdict(s))
    store["items"] = items
    _atomic_write(store)
    return s

def delete_schedule(sid: int) -> bool:
    store = _read_store()
    n0 = len(store["items"])
    store["items"] = [it for it in store["items"] if it.get("id") != sid]
    changed = len(store["items"]) != n0
    if changed:
        _atomic_write(store)
    return changed

def get_by_id(sid: int) -> Optional[Dict[str, Any]]:
    store = _read_store()
    for it in store.get("items", []):
        if it.get("id") == sid:
            return it
    return None

def list_month(year: int, month: int) -> Dict[int, List[Dict[str, Any]]]:
    """{ day:int : [item, ...] }"""
    store = _read_store()
    result: Dict[int, List[Dict[str, Any]]] = {}
    for it in store.get("items", []):
        ms = it.get("meeting_start")
        try:
            dt = datetime.fromisoformat(ms)
        except Exception:
            # 허용 형식: "YYYY-MM-DDTHH:MM:SS"만 사용
            continue
        if dt.year == year and dt.month == month:
            result.setdefault(dt.day, []).append(it)
    # 일자별 시간순 정렬
    for d in result:
        result[d].sort(key=lambda x: x.get("meeting_start", ""))
    return result

def list_day(yyyy_mm_dd: str) -> List[Dict[str, Any]]:
    y, m, d = map(int, yyyy_mm_dd.split("-"))
    by_day = list_month(y, m)
    return by_day.get(d, [])
