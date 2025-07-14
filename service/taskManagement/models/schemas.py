from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    user_input: str

class ChatResponse(BaseModel):
    convo_id: str
    response: str