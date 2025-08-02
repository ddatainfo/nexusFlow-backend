import asyncio
from uuid import uuid4
from utils.llm import call_mistral
from utils.extractor import extract_fields, is_title_too_vague
from agents.ticket_analysis import TicketAnalysisAgent
from models.schemas import ChatResponse

async def handle_ticket_fields(user_input: str, state: dict, conversation: list, convo_id: str) -> str:
    # Handle vague or missing title
    if state.get("awaiting_title"):
        title_candidate = user_input.strip()
        vague_check_prompt = (
            f'The user entered the issue title: "{title_candidate}".\n'
            "Is this title vague or acceptable?\n"
            "Respond strictly with one of the following:\n"
            "- VAGUE\n"
            "- OK"
        )
        result = call_mistral(vague_check_prompt).strip().upper()
        retry_count = state.get("vague_retry_count", 0)

        if result == "VAGUE":
            if retry_count < 1:
                state["vague_retry_count"] = retry_count + 1
                state["last_vague_title"] = title_candidate
                clarification_prompt = (
                    f"The user entered this vague issue title: \"{title_candidate}\".\n"
                    "Please rewrite ONE sentence that asks the user to provide a clearer title.\n"
                    "DO NOT include greetings like 'Hi', 'Hello', 'Dear User'.\n"
                    "DO NOT include sign-offs like 'Thanks', 'Regards'.\n"
                    "Ask the question plainly and directly.\n"
                    "Speak like a helpful assistant chatbot, not an email.\n"
                    "Example output: Could you please provide a more specific title for this issue?\n"
                    "Respond with ONE plain sentence only."
                )
                response = call_mistral(clarification_prompt)
                if any(bad in response.lower() for bad in ["dear", "regards", "hope this", "thank you"]):
                    print("Mistral generated email-style response. Using fallback.")
                    response = "Could you please provide a more specific title for this issue?"
                conversation.append({"role": "assistant", "content": response})
                print(f"Starlistant: {response}")
                return response
            else:
                if state.get("last_vague_title") == title_candidate:
                    print("User repeated the same vague title. Accepting it.")
                else:
                    print("Second title is still vague. Accepting it anyway.")

        state["fields"]["title"] = title_candidate
        state["awaiting_title"] = False

        if not state["fields"].get("description"):
            # FIXED: Instead of generating a description, ask the user for details - CRISP VERSION
            description_prompt = (
                f'The user provided the title: "{title_candidate}".\n'
                "Ask them for more details about this issue in ONE short sentence. "
                "Keep it brief and direct. "
                "Example: 'Please describe what exactly happens.'\n"
                "Respond with ONE short sentence only."
            )
            response = call_mistral(description_prompt)
            
            # Fallback if Mistral fails - CRISP VERSION
            if not response or any(bad in response.lower() for bad in ["dear", "regards", "hope this"]) or len(response) > 80:
                response = "Please describe what exactly happens."
            
            state["awaiting_description"] = True
            conversation.append({"role": "assistant", "content": response})
            print(f"Starlistant: {response}")
            return response
        else:
            print("Description already exists. Skipping description prompt.")

    # Handle description input and ask for priority
    if state.get("awaiting_description"):
        state["fields"]["description"] = user_input.strip()
        state["awaiting_description"] = False

        if state["fields"].get("priority"):
            print("\n📝 Ticket captured successfully:")
            print(f"Title      : {state['fields']['title']}")
            print(f"Description: {state['fields']['description']}")
            print(f"Priority   : {state['fields']['priority']}")
            print("-" * 50)

            agent = TicketAnalysisAgent(convo_id)
            result_msg = agent.analyze_ticket()
            print(f"\n📣 {result_msg} ✅")
            return state["conversation"][-1]["content"]

        priority_prompt = (
            "Ask the user to choose a priority level for this issue.\n"
            "Respond ONLY in this exact format using line breaks and tags:\n"
            "What is the priority of this issue?\n"
            "b$low$b\n"
            "b$medium$b\n"
            "b$high$b\n"
            "DO NOT include descriptions, explanations, or any other text."
            "DO NOT include greetings, sign-offs, or examples."
        )
        response = call_mistral(priority_prompt)
        state["awaiting_priority"] = True
        conversation.append({"role": "assistant", "content": response})
        print(f"Starlistant: {response}")
        return response

    if state.get("awaiting_priority"):
        # Normalize button input like b$low$b
        if user_input.startswith("b$") and user_input.endswith("$b"):
            user_input = user_input[2:-2].strip()

        priority = user_input.strip().capitalize()
        if priority not in ["Low", "Medium", "High"]:
            response = "Priority must be Low, Medium, or High. Please enter a valid priority."
            conversation.append({"role": "assistant", "content": response})
            print(f"Starlistant: {response}")
            return response

        state["fields"]["priority"] = priority
        state["awaiting_priority"] = False
        
        print("\n📝 Ticket captured successfully:")
        print(f"Title      : {state['fields']['title']}")
        print(f"Description: {state['fields']['description']}")
        print(f"Priority   : {state['fields']['priority']}")
        print("-" * 50)

        agent = TicketAnalysisAgent(convo_id)
        result_msg = agent.analyze_ticket()
        print(f"\n📣 {result_msg} ✅")
        return state["conversation"][-1]["content"]

    # Check if all fields are completed in one message
    completed = extract_fields(user_input, state)
    if completed:
        print("\n📝 Ticket captured successfully (from one message):")
        print(f"Title      : {state['fields']['title']}")
        print(f"Description: {state['fields']['description']}")
        print(f"Priority   : {state['fields']['priority']}")
        print("--------------------------------------------------")

        analysis_agent = TicketAnalysisAgent(convo_id)
        result = analysis_agent.analyze_ticket()
        state["awaiting_ticket_confirmation"] = True
        return state["conversation"][-1]["content"]

    # Check for missing fields and prompt for the first one
    for field in ["title", "description", "priority"]:
        if not state["fields"].get(field):
            state[f"awaiting_{field}"] = True

            if field == "title":
                mistral_prompt = (
                    "The user has not provided the issue title yet.\n"
                    "Ask them to provide a clear and specific title.\n"
                    "Only respond with a single, polite sentence asking for the title.\n"
                    "No greetings, no sign-offs.\n"
                    "Example: 'Could you please provide the title of the issue you're facing?'"
                )

            elif field == "description":
                # FIXED: Ask for more details instead of restating the problem - CRISP VERSION
                mistral_prompt = (
                    f'The user already provided this title: "{state["fields"]["title"]}".\n'
                    "Ask them for more details about this issue in ONE short sentence. "
                    "Keep it brief and direct. "
                    "Example: 'Please describe what exactly happens.'\n"
                    "Respond with ONE short sentence only."
                )

            elif field == "priority":
                mistral_prompt = (
                    "Ask the user to choose a priority level for this issue.\n"
                    "Respond ONLY in this exact format using line breaks and tags:\n"
                    "What is the priority of this issue?\n"
                    "b$low$b\n"
                    "b$medium$b\n"
                    "b$high$b\n"
                    "DO NOT include descriptions, explanations, or any other text."
                    "DO NOT include greetings, sign-offs, or examples."
                )

            prompt = call_mistral(mistral_prompt).strip()

            # Safety fallback if Mistral fails or returns junk
            if not prompt or any(bad in prompt.lower() for bad in ["dear", "regards", "thank", "hope"]):
                prompt = {
                    "title": "Could you please provide the title of the issue you're facing?",
                    "description": "Please describe what exactly happens.",
                    "priority": "\n".join([
                        "What is the priority of this issue?",
                        "b$low$b",
                        "b$medium$b",
                        "b$high$b"
                    ])
                }[field]

            conversation.append({"role": "assistant", "content": prompt})
            print(f"Starlistant: {prompt}")
            return prompt

    return None