import re
from utils.llm import call_mistral

def is_greeting_only(text: str) -> bool:
    cleaned = re.sub(r'[^\w\s]', '', text.lower()).strip()
    pattern = r"^(hi+|hello+|hey+|yo+|greetings|good\s(morning|evening|afternoon))( there| team| everyone)?$"
    return re.match(pattern, cleaned) is not None

def handle_greeting(user_input: str, state: dict, conversation: list) -> bool:
    if is_greeting_only(user_input) and not state.get("greeted"):
        state["greeted"] = True
        greeting_prompt = (
            f"The user said: \"{user_input.strip()}\".\n"
            "Reply only with a short, one-sentence greeting. Do not say anything extra like well wishes or encouragement. Strictly end the sentence after offering help"
        )
        response = call_mistral(greeting_prompt)
        conversation.append({"role": "assistant", "content": response})
        print(f"Starlistant: {response}")
        return True
    return False
