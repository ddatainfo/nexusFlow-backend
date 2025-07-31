from pydantic import BaseModel
from typing import Optional

# class ChatRequest(BaseModel):
#     user_input: str

# class ChatResponse(BaseModel):
#     convo_id: str
#     response: str

class ChatRequest(BaseModel):
    user_input: str
    query_type: str  # New field for dropdown selection ("tickets_base" or "knowledge_base")

class ChatResponse(BaseModel):
    convo_id: str
    response: str