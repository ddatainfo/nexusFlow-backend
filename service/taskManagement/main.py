from fastapi import FastAPI, Request, File, Form, UploadFile
from fastapi.responses import JSONResponse
from models.schemas import ChatRequest, ChatResponse
from handlers.greeting import handle_greeting
from handlers.ticket import handle_ticket_fields
from handlers.attachment_handler import handle_attachment
from utils.state import init_conversation, get_conversation_id, save_conversation_state, conversation_states
from utils.extractor import is_issue_related, check_field_update_intent
from utils.llm import call_mistral
from typing import Optional
import re
import uuid

app = FastAPI(title="JIRA Chatbot API")

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: Request,
    user_input: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    user_input = user_input.strip()

    # ✅ Try to reuse convo_id from cookie
    convo_id = request.cookies.get("convo_id")
    print(f"Received convo_id: {convo_id}")

    if not convo_id or convo_id not in conversation_states:
        convo_id = str(uuid.uuid4())
        init_conversation(convo_id)
        save_conversation_state()

    # ✅ State must be loaded before usage
    state = conversation_states[convo_id]
    conversation = state["conversation"]
    
    # Handle greeting
    if handle_greeting(user_input, state, conversation):
        response = conversation[-1]["content"]
        save_conversation_state()
        res = JSONResponse(content=ChatResponse(convo_id=convo_id, response=response).dict())
        res.set_cookie("convo_id", convo_id)
        return res

    # Handle ticket creation confirmation
    if state.get("awaiting_ticket_confirmation"):
        cleaned_input = re.sub(r"b\$|\$b", "", user_input.lower())
        conversation.append({"role": "user", "content": user_input})

        if cleaned_input in ["yes", "create"]:
            state["awaiting_ticket_confirmation"] = False
            state["awaiting_attachment_confirmation"] = True
            response = "Would you like to upload any image, PDF, video, or link with this ticket? (yes/no)"
            conversation.append({"role": "assistant", "content": response})
            print(f"Starlistant: {response}")
            save_conversation_state()
            res = JSONResponse(content=ChatResponse(convo_id=convo_id, response=response).dict())
            res.set_cookie("convo_id", convo_id)
            return res
        elif cleaned_input in ["no"]:
            state["awaiting_ticket_confirmation"] = False
            response = "Okay, I won't create a ticket. Let me know if you need anything else."
            conversation.append({"role": "assistant", "content": response})
            print(f"Starlistant: {response}")
            new_convo_id = str(uuid.uuid4())
            init_conversation(new_convo_id)
            save_conversation_state()
            res = JSONResponse(content=ChatResponse(convo_id=new_convo_id, response=response).dict())
            res.set_cookie("convo_id", new_convo_id)
            return res

    # Handle attachment confirmation or upload
    if state.get("awaiting_attachment_confirmation") or state.get("awaiting_file_upload"):
        response = await handle_attachment(user_input, file, state, conversation, convo_id)
        if response:
            return response
        else:
            print("[WARN] handle_attachment returned None but state was still awaiting attachment")

    # Add user input to conversation
    conversation.append({"role": "user", "content": user_input})
    save_conversation_state()

    # Handle ticket fields
    response = await handle_ticket_fields(user_input, state, conversation, convo_id)
    if response:
        save_conversation_state()
        res = JSONResponse(content=ChatResponse(convo_id=convo_id, response=response).dict())
        res.set_cookie("convo_id", convo_id)
        return res

    # Check for field update intent
    field_to_update = check_field_update_intent(user_input)
    if field_to_update:
        state["awaiting_field_update"] = field_to_update
        response = f"Sure! Please tell me the new {field_to_update}."
        conversation.append({"role": "assistant", "content": response})
        print(f"Starlistant: {response}")
        save_conversation_state()
        res = JSONResponse(content=ChatResponse(convo_id=convo_id, response=response).dict())
        res.set_cookie("convo_id", convo_id)
        return res

    # Handle missing fields if issue-related
    if is_issue_related(user_input) and not all(state["fields"].values()):
        state["awaiting_title"] = True
        clarification_prompt = (
            f"The user said: \"{user_input.strip()}\".\n"
            "Your job is to *politely ask* the user to provide a short and specific title for their issue.\n"
            "Do NOT suggest or generate any title yourself.\n"
            "Only respond with one polite question asking the user to provide the title.\n"
            "Keep it short and professional.\n"
            "Example response: 'Could you please provide the title of the issue you’re facing?'"
        )
        response = call_mistral(clarification_prompt)
        conversation.append({"role": "assistant", "content": response})
        print(f"Starlistant: {response}")
        save_conversation_state()
        res = JSONResponse(content=ChatResponse(convo_id=convo_id, response=response).dict())
        res.set_cookie("convo_id", convo_id)
        return res

    # Handle unrelated input
    professional_prompt = (
        f'The user mentioned: "{user_input.strip()}".\n'
        "Respond professionally, but do not ask for title, description, or priority unless an issue is mentioned."
    )
    response = call_mistral(professional_prompt)
    conversation.append({"role": "assistant", "content": response})
    save_conversation_state()
    print(f"Starlistant: {response}")
    res = JSONResponse(content=ChatResponse(convo_id=convo_id, response=response).dict())
    res.set_cookie("convo_id", convo_id)
    return res

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
