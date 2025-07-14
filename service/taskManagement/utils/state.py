import uuid
from fastapi import Request
from .db import save_conversation_to_db, load_conversation_from_db

conversation_states = {}

def init_conversation(convo_id: str):
    db_state = load_conversation_from_db(convo_id)
    if db_state:
        conversation_states[convo_id] = db_state
    else:
        conversation_states[convo_id] = {
            "fields": {"title": None, "description": None, "priority": None},
            "conversation": [],
            "awaiting_title": False,
            "awaiting_description": False,
            "awaiting_priority": False,
            "awaiting_ticket_confirmation": False,
            "awaiting_attachment_confirmation": False,
            "awaiting_file_upload": False,
            "awaiting_field_update": None,
            "greeted": False,
            "vague_retry_count": 0,
            "last_vague_title": None
        }

def save_conversation_state():
    for convo_id, state in conversation_states.items():
        save_conversation_to_db(convo_id, state)

def get_conversation_id(request: Request) -> str:
    convo_id = request.cookies.get("conversation_id")
    if not convo_id:
        convo_id = str(uuid.uuid4())
        print(f"🆕 New session started: {convo_id}")
    return convo_id