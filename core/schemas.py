# core/schemas.py
from typing import List, Optional
from pydantic import BaseModel, Field

class ActionItem(BaseModel):
    title: str = Field(..., description="행동을 설명하는 한 줄")
    owner: Optional[str] = Field(None, description="담당자(없으면 None)")
    due: Optional[str] = Field(None, description="YYYY-MM-DD 또는 None")
    priority: Optional[str] = Field("M", description="H/M/L")
    status: Optional[str] = Field("todo", description="todo/in-progress/done")
    notes: Optional[str] = None

class Decision(BaseModel):
    text: str
    impact: Optional[str] = None  # cost/scope/schedule 등

class Issue(BaseModel):
    text: str
    severity: Optional[str] = "M"

class MeetingSummary(BaseModel):
    meeting_title: str
    date: Optional[str] = None
    participants: List[str] = []
    highlevel_summary: List[str]
    key_points: List[str]
    decisions: List[Decision] = []
    action_items: List[ActionItem] = []
    issues: List[Issue] = []
    risks: List[str] = []
    metrics: List[str] = []  # 예: "모델 정확도 87%→90% 목표"
