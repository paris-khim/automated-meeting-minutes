from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class ActionItem(BaseModel):
    assignee: str = Field(..., description="The person responsible for the task")
    task: str = Field(..., description="The specific task description")
    deadline: Optional[datetime] = None

class MeetingInsight(BaseModel):
    summary: str = Field(..., description="Executive summary of the meeting")
    decisions: List[str] = Field(default_factory=list, description="List of key decisions made")
    risks: List[str] = Field(default_factory=list, description="Strategic or technical risks identified")
    action_items: List[ActionItem] = Field(default_factory=list)

class TranscribedSegment(BaseModel):
    speaker: str
    text: str
    start_time: float
    end_time: float
