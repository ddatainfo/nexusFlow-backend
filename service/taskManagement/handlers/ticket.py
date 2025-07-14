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
            description_prompt = (
                f'The user gave this issue title: "{title_candidate}".\n'
                "Ask the user to describe the issue clearly.\n"
                "One sentence only.\n"
                "No greetings, examples, or sign-offs.\n"
                "Example: 'Could you describe the issue in more detail?'"
            )
            response = call_mistral(description_prompt)
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
            f'The user described the issue as: "{user_input.strip()}".\n'
            "Ask them to choose a priority: Low, Medium, or High.\n"
            "Only one sentence.\n"
            "No explanation of what priority means.\n"
            "No greetings or sign-offs.\n"
            "Example: 'What is the priority of this issue? (Low, Medium, or High)'"
        )
        response = call_mistral(priority_prompt)
        state["awaiting_priority"] = True
        conversation.append({"role": "assistant", "content": response})
        print(f"Starlistant: {response}")
        return response

    if state.get("awaiting_priority"):
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

    return None