import os
import uuid
from fastapi import UploadFile
from fastapi.responses import JSONResponse
from models.schemas import ChatResponse
from services.jira import create_jira_ticket
from utils.state import save_conversation_state, conversation_states, init_conversation


async def handle_attachment(user_input: str, file: UploadFile, state: dict, conversation: list, convo_id: str) -> JSONResponse:
    cleaned_input = user_input.lower().strip()
    conversation.append({"role": "user", "content": user_input})

    # ✅ Step 1: Handle actual file or text link submission (highest priority if waiting for attachment)
    if state.get("awaiting_file_upload"):
        state["awaiting_file_upload"] = False  # Reset upload expectation
        attachment = None

        if file:
            # file_location = f"./uploads/{file.filename}"
            # with open(file_location, "wb") as f:
            #     f.write(await file.read())
            upload_dir = "./uploads"
            os.makedirs(upload_dir, exist_ok=True)
            file_location = os.path.join(upload_dir, file.filename)
            with open(file_location, "wb") as f:
                f.write(await file.read())

            print(f"📁 File saved to: {file_location}")
            print(f"📦 File size: {os.path.getsize(file_location)} bytes")
            print(f"📄 File content type: {file.content_type}")

            attachment = {
                "path": file_location,
                "filename": file.filename,
                "content_type": file.content_type
            }

            conversation.append({"role": "user", "content": f"[Uploaded file: {file.filename}]"})
        elif user_input.strip():
            attachment = user_input.strip()
            conversation.append({"role": "user", "content": f"[Attachment description: {attachment}]"})
        else:
            state["file_retry_count"] = state.get("file_retry_count", 0) + 1
            if state["file_retry_count"] < 2:
                state["awaiting_file_upload"] = True
                response = "⚠️ I didn't receive any file or attachment description. Please upload a file or provide a link/description:"
                conversation.append({"role": "assistant", "content": response})
                save_conversation_state()
                res = JSONResponse(content=ChatResponse(convo_id=convo_id, response=response).dict())
                res.set_cookie("convo_id", convo_id)
                return res
            else:
                attachment = None  # Give up and continue without attachment

        # ✅ Create the ticket with the provided attachment
        ticket_key, ticket_url = create_jira_ticket(state["fields"], attachment=attachment)
        if ticket_key:
            response = (
                f"Your ticket with the attachment has been created successfully ✅\n"
                f"🎫 Ticket Key: {ticket_key}\n"
                f"🔗 Link: {ticket_url}"
            )
        else:
            response = "❌ Ticket creation failed. Please try again later."

        conversation.append({"role": "assistant", "content": response})
        new_convo_id = str(uuid.uuid4())
        init_conversation(new_convo_id)
        save_conversation_state()
        res = JSONResponse(content=ChatResponse(convo_id=new_convo_id, response=response).dict())
        res.set_cookie("convo_id", new_convo_id)
        return res

    # ✅ Step 2: User says YES to attachment — prompt for upload
    elif cleaned_input in ["yes", "y", "sure", "ok"]:
        state["awaiting_attachment_confirmation"] = False
        state["awaiting_file_upload"] = True
        response = "Please paste the file path, link, or describe the attachment you'd like to upload:"
        conversation.append({"role": "assistant", "content": response})
        save_conversation_state()
        res = JSONResponse(content=ChatResponse(convo_id=convo_id, response=response).dict())
        res.set_cookie("convo_id", convo_id)
        return res

    # ✅ Step 3: User says NO — skip attachments and create ticket directly
    elif cleaned_input in ["no", "n"]:
        state["awaiting_attachment_confirmation"] = False
        ticket_key, ticket_url = create_jira_ticket(state["fields"], attachment=None)
        if ticket_key:
            response = (
                f"Your ticket has been created successfully ✅ (no attachment)\n"
                f"🎫 Ticket Key: {ticket_key}\n"
                f"🔗 Link: {ticket_url}"
            )
        else:
            response = "❌ Failed to create ticket. Please try again later."

        conversation.append({"role": "assistant", "content": response})
        new_convo_id = str(uuid.uuid4())
        init_conversation(new_convo_id)
        save_conversation_state()
        res = JSONResponse(content=ChatResponse(convo_id=new_convo_id, response=response).dict())
        res.set_cookie("convo_id", new_convo_id)
        return res

    # ❓ Fallback: Invalid input
    response = "❓ I didn't understand that. Please respond with 'yes' or 'no'."
    conversation.append({"role": "assistant", "content": response})
    save_conversation_state()
    res = JSONResponse(content=ChatResponse(convo_id=convo_id, response=response).dict())
    res.set_cookie("convo_id", convo_id)
    return res
