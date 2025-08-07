from typing import Optional, List
from pydantic import BaseModel

class JiraTaskItem(BaseModel):
    key: str
    summary: Optional[str]
    status: Optional[str]
    created: Optional[str]

class JiraTaskListResponse(BaseModel):
    tasks: List[JiraTaskItem]

class ChatRequest(BaseModel):
    user_input: str
    user_id: Optional[int] = 7