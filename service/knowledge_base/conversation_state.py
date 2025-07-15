import pickle
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from fastapi import HTTPException, Request
import logging

logger = logging.getLogger(__name__)

CONVERSATION_STATE_FILE = "/home/ddata/Interns_work/AI_chatbot/chatbot/dev_conversation_state.pkl"

# Global state storage for multiple conversations
conversation_states = {}

def load_conversation_state() -> dict:
    try:
        if Path(CONVERSATION_STATE_FILE).exists():
            with open(CONVERSATION_STATE_FILE, "rb") as f:
                state = pickle.load(f)
                logger.info("Conversation state loaded from disk")
                # Clean up stale conversations (older than 24 hours)
                cutoff = datetime.now() - timedelta(hours=24)
                for cid in list(state.keys()):
                    if state[cid].get("last_updated", datetime.min) < cutoff:
                        del state[cid]
                return state
        return {}
    except Exception as e:
        logger.error(f"Failed to load conversation state: {str(e)}")
        return {}

def save_conversation_state():
    try:
        for state in conversation_states.values():
            state["last_updated"] = datetime.now()
        with open(CONVERSATION_STATE_FILE, "wb") as f:
            pickle.dump(conversation_states, f)
        logger.info("Conversation state saved to disk")
    except Exception as e:
        logger.error(f"Failed to save conversation state: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save conversation state: {str(e)}")

def init_conversation_state(conversation_id: str):
    conversation_states[conversation_id] = {
        "context_history": [],
        "greeted": False,
        "ticket_started": False,
        "awaiting_attachment": False,
        "awaiting_upload_confirmation": False,
        "similar_checked": False,
        "awaiting_ticket_confirmation": False,
        "awaiting_validation_fix": False,
        "awaiting_no_similarity_confirmation": False,
        "pending_fields": {
            "title": None,
            "issue": None,
            "priority": None,
            "component": None,
            "issuetype": None,
            "attachments": []
        },
        "finalized_ticket": None,
        "jira_response": None,
        "last_updated": datetime.now()
    }
    logger.info(f"Initialized conversation state for ID: {conversation_id}")
    save_conversation_state()

def get_conversation_id(request: Request) -> str:
    if "conversation_id" not in request.session:
        conversation_id = str(uuid.uuid4())
        request.session["conversation_id"] = conversation_id
        init_conversation_state(conversation_id)
        logger.info(f"New conversation ID generated: {conversation_id}")
    return request.session["conversation_id"]

def update_chat_history(conversation_id: str, question: str, answer: str):
    state = conversation_states.get(conversation_id)
    if state is not None:
        history = state.setdefault("chat_history", [])
        history.append({"question": question, "answer": answer})
        # Keep only last 5
        if len(history) > 5:
            state["chat_history"] = history[-5:]
        else:
            state["chat_history"] = history
        save_conversation_state()