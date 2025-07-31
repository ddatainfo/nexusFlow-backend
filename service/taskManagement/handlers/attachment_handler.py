from fastapi import UploadFile
from fastapi.responses import JSONResponse
from services.jira import create_jira_ticket
from models.schemas import ChatResponse
from utils.state import persist_conversation, conversation_states
from typing import Optional
import logging
import re
import os

logger = logging.getLogger(__name__)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

async def handle_attachment(user_input: str, file: Optional[UploadFile], state: dict, conversation: list, convo_id: str) -> JSONResponse:
    cleaned_input = user_input.strip().lower()

    # Step 1: Confirm upload intent
    if state.get("awaiting_attachment_confirmation"):
        if cleaned_input in ["yes", "y", "ok", "sure"]:
            state["awaiting_attachment_confirmation"] = False
            state["awaiting_file_upload"] = True
            response = "📎 Great! Please upload a file or paste a link to proceed. $upload$"
            conversation.append({"role": "assistant", "content": response})
            persist_conversation(convo_id, state)
            return JSONResponse(content=ChatResponse(convo_id=convo_id, response=response).dict())

        elif cleaned_input in ["no", "n"]:
            state["awaiting_attachment_confirmation"] = False
            try:
                ticket_key, ticket_url = create_jira_ticket(state["fields"], attachment=None)
                response = (
                    f"✅ Ticket created without attachment.<br/>"
                    f"🎫 Ticket Key: {ticket_key}<br/>"
                    f"🔗 Link: <a href='{ticket_url}' target='_blank'>{ticket_url}</a>"
                ) if ticket_key else "❌ Ticket creation failed."

            except Exception as e:
                logger.error(f"❌ Ticket creation failed without attachment: {str(e)}")
                response = "❌ Ticket creation failed. Please try again later."

            conversation.append({"role": "assistant", "content": response})
            conversation_states.pop(convo_id, None)
            return JSONResponse(content=ChatResponse(convo_id=convo_id, response=response).dict())

    # Step 2: Handle file upload
    if file and state.get("awaiting_file_upload"):
        try:
            save_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(save_path, "wb") as f:
                f.write(await file.read())

            state["attachment"] = {
                "filename": file.filename,
                "path": save_path,
                "content_type": file.content_type
            }
            state.update({
                "ready_to_submit": True,
                "awaiting_file_upload": False,
                "awaiting_attachment_confirmation": False
            })
            persist_conversation(convo_id, state)

        except Exception as e:
            logger.error(f"❌ File upload failed: {str(e)}")
            response = "❌ Failed to upload file. Please try again."
            conversation.append({"role": "assistant", "content": response})
            return JSONResponse(content=ChatResponse(convo_id=convo_id, response=response).dict())

    # Step 3: Handle pasted link
    elif user_input and re.match(r"^https?://", user_input) and state.get("awaiting_file_upload"):
        state["attachment"] = user_input.strip()
        state.update({
            "ready_to_submit": True,
            "awaiting_file_upload": False,
            "awaiting_attachment_confirmation": False
        })
        persist_conversation(convo_id, state)

    # Step 4: Auto-submit ticket if ready
    if state.get("ready_to_submit") and state.get("fields", {}).get("title"):
        logger.info("🚀 Creating Jira ticket after upload/link...")
        try:
            fields = state["fields"]
            attachment = state.get("attachment")
            ticket_key, ticket_url = create_jira_ticket(fields, attachment=attachment)
            # ✅ Save ticket info in state for external use (upload endpoint)
            state["ticket_key"] = ticket_key
            state["ticket_url"] = ticket_url

            if isinstance(attachment, dict):
                attach_note = f"📎 Attachment uploaded: {attachment['filename']}"
            elif isinstance(attachment, str):
                attach_note = f"🔗 Link attached: {attachment}"
            else:
                attach_note = ""

            response = (
                f"✅ Ticket created successfully!<br/>"
                f"🎫 Ticket Key: {ticket_key}<br/>"
                f"🔗 Link: <a href='{ticket_url}' target='_blank'>{ticket_url}</a><br/>"
                f"{attach_note}"
            ) if ticket_key else "❌ Ticket creation failed."

            # Clean up
            state.update({
                "ready_to_submit": False,
                "ticket_flow_started": False,
                "awaiting_file_upload": False,
                "awaiting_attachment_confirmation": False,
            })
            persist_conversation(convo_id, state)
            conversation.append({"role": "assistant", "content": response})
            return JSONResponse(content=ChatResponse(convo_id=convo_id, response=response).dict())

        except Exception as e:
            logger.error(f"❌ Ticket creation failed after upload/link: {str(e)}")
            response = "❌ Ticket creation failed. Please try again."
            conversation.append({"role": "assistant", "content": response})
            return JSONResponse(content=ChatResponse(convo_id=convo_id, response=response).dict())

    # Step 5: If awaiting upload but nothing valid was given
    if state.get("awaiting_file_upload"):
        response = "⚠️ No file or valid link detected. Please upload or paste a link."
        conversation.append({"role": "assistant", "content": response})
        return JSONResponse(content=ChatResponse(convo_id=convo_id, response=response).dict())

    # Step 6: Fallback
    response = "⚠️ Please upload a file or paste a valid link."
    conversation.append({"role": "assistant", "content": response})
    return JSONResponse(content=ChatResponse(convo_id=convo_id, response=response).dict())