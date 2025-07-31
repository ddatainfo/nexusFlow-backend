from fastapi import Request, UploadFile
from fastapi.responses import JSONResponse
from models.schemas import ChatResponse
from utils.state import *
from utils.ticket import handle_ticket_flow
from handlers.greeting import handle_greeting
from handlers.attachment_handler import handle_attachment
from utils.extractor import is_issue_related
import re

async def handle_chat(request: Request, user_input: str, file: UploadFile = None) -> JSONResponse:
    convo_id = get_conversation_id(request)
    if convo_id not in conversation_states:
        init_conversation(convo_id)
        save_conversation_state()

    state = conversation_states[convo_id]
    conversation = state["conversation"]

    # Log user message
    conversation.append({"role": "user", "content": user_input})

    # Greeting
    if handle_greeting(user_input, state, conversation):
        save_conversation_state()
        return JSONResponse(content=ChatResponse(convo_id=convo_id, response=conversation[-1]["content"]).dict())


    # Upload handler
    if state.get("awaiting_ticket_confirmation") or state.get("awaiting_attachment_confirmation") or state.get("awaiting_file_upload"):
        return await handle_attachment(user_input, file, state, conversation, convo_id)

    # Ticket fields handler
    return await handle_ticket_flow(convo_id, user_input)



# # THIS is chathandlers /chat_handler.py
# from fastapi import Request, UploadFile
# from fastapi.responses import JSONResponse
# from models.schemas import ChatResponse
# from utils.state import *
# from utils.ticket import handle_ticket_flow
# from handlers.greeting import handle_greeting
# from handlers.attachment_handler import handle_attachment
# import re

# async def handle_chat(request: Request, user_input: str, file: UploadFile = None) -> JSONResponse:
#     convo_id = get_conversation_id(request)
#     if convo_id not in conversation_states:
#         init_conversation(convo_id)
#         save_conversation_state()
#     state = conversation_states[convo_id]

#     # --- 🧠 Route based on conversation state ---
#     if is_greeting_only(user_input):
#         return await handle_greeting(convo_id, user_input)
    
#     if state.get("awaiting_ticket_confirmation") or state.get("awaiting_attachment_confirmation") or state.get("awaiting_file_upload"):
#         return await handle_upload(convo_id, user_input, file)

#     return await handle_ticket_flow(convo_id, user_input)