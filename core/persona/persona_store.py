# -*- coding: utf-8 -*-
# core/persona/persona_store.py
"""
설문 기반 페르소나 프로필 저장소

- key: speaker_id (예: "speaker_01")
- 값: system prompt, voice 옵션, backend 설정 등 정적 프로필
"""

import json
import os
import copy
from typing import Dict, Optional, Any

PERSONA_PATH = os.path.join("data", "persona", "prompts.json")


def _deep_merge(dst: dict, src: dict) -> dict:
    out = copy.deepcopy(dst)
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class PersonaStore:
    def __init__(self, path: str = PERSONA_PATH):
        self.path = path
        self.data: Dict[str, Any] = {"default_style": {}}
        self._load()

    # --------------------------------------------------------------
    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    # --------------------------------------------------------------
    def get(self, name: str) -> Dict[str, Any]:
        return self.data.get(name, {})

    def upsert(self, name: str, persona_dict: Dict[str, Any], merge: bool = True):
        """
        name = speaker_id (예: "speaker_01")
        설문/이력 정보를 병합 저장.
        """
        if merge and name in self.data:
            self.data[name] = _deep_merge(self.data[name], persona_dict or {})
        else:
            self.data[name] = persona_dict or {}
        self.save()

    # --------------------------------------------------------------
    # system prompt / backend 선택
    # --------------------------------------------------------------
    def build_system_prompt(self, name: Optional[str]) -> str:
        """
        name: speaker_id 또는 None
        """
        default = self.data.get("default_style", {}).get("system", "")
        if not name:
            return default or "You are a helpful assistant."

        p = self.data.get(name, {})
        sys = p.get("system") or default or "You are a helpful assistant."

        # 지원 언어 정보
        if p.get("voice", {}).get("languages"):
            langs = ", ".join(p["voice"]["languages"])
            sys += f"\n(지원 언어: {langs})"

        # 기본 가이드라인
        if "guidelines" in self.data.get("default_style", {}):
            gl = "\n".join(f"- {g}" for g in self.data["default_style"]["guidelines"])
            sys += f"\n\n[Guidelines]\n{gl}"

        return sys

    def choose_backend(self, name: Optional[str]) -> str:
        p = self.data.get(name or "", {})
        return p.get("backend", "openai:gpt-4o-mini")

    # --------------------------------------------------------------
    # 설문 → JSON 병합
    # --------------------------------------------------------------
    def update_from_survey(self, name: str, survey: Dict[str, Any]):
        """
        survey 예:
        {
          "tone": "정중/직설/친근/데이터중심",
          "summary_format": "개조식/서술식/키워드/표",
          "keywords": ["ASAP","애자일"],
          "alarm": "회의 30분 전",
          "report_focus": "정확성/간결성/시각화",
          "backend": "openai:gpt-4o-mini"
        }

        name = speaker_id (예: "speaker_01")
        """
        patch = {
            "system": (
                f"너는 '{name}' 개인 비서다. 말투는 {survey.get('tone', '명확/직설')}."
                f" 보고서는 {survey.get('summary_format', '개조식, 결론 우선')} 구조를 기본으로 한다."
                " 협업을 중시하고 상대의 의견을 존중하며, 일정/할일은 우선순위와 마감일을 강조한다."
                " 답변은 간결하고 정확하며 필요시 도표·대시보드·체크리스트·나열식을 사용한다."
                " 꼼꼼하고 체계적인 느낌을 유지한다."
            ),
            "voice": {
                "tone": survey.get("tone", "명확/직설 + 꼼꼼"),
                "summary_format": survey.get("summary_format", "개조식, 결론 우선"),
                "languages": ["한국어", "영어"],
            },
            "favorites": {
                "alarm": survey.get("alarm", "회의 30분 전"),
                "report_focus": survey.get(
                    "report_focus", "정확성, 간결성, 데이터 기반"
                ),
                "style": "도표/체크리스트/대시보드 가능",
            },
            "keywords": survey.get("keywords", []),
        }

        if survey.get("backend"):
            patch["backend"] = survey["backend"]

        self.upsert(name, patch, merge=True)
