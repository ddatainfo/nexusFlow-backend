import re
from utils.llm import call_mistral  # Adjust path as per your structure

def is_greeting_only(text: str) -> bool:
    cleaned = re.sub(r'[^\w\s]', '', text.lower()).strip()
    pattern = r"^(hi+|hello+|hey+|yo+|greetings|good\s(morning|evening|afternoon))( there| team| everyone)?$"
    return re.match(pattern, cleaned) is not None

def is_issue_related(user_input):
    prompt = (
        "Determine whether the following message is reporting an issue, bug, error, or malfunction of any kind.\n"
        "Only reply with 'Yes' or 'No'.\n\n"
        "Examples:\n"
        "- 'My Jira ticket is not creating.' → Yes\n"
        "- 'Slack messages are not sending.' → Yes\n"
        "- 'Just wanted to say hi.' → No\n"
        "- 'Do you support dark mode?' → No\n\n"
        f"Message: \"{user_input.strip()}\"\n"
        "Is this reporting a problem?"
    )
    try:
        response = call_mistral(prompt).strip().lower()
        if "yes" in response:
            return True
        if "no" in response:
            return False
    except Exception as e:
        print("⚠️ call_mistral failed in is_issue_related:", e)

    return False

def format_response(status: str, message: str = "", data: dict = None) -> dict:
    return {
        "status": status,
        "message": message,
        "data": data or {}
    }
