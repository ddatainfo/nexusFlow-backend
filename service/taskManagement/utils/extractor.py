import re
import spacy
from utils.llm import call_mistral

nlp = spacy.load("en_core_web_sm")

def extract_fields(user_input, state):
    print(f"\n🌐 Extracting fields from: '{user_input}'")
    lowered = user_input.lower().strip()
    fields = state["fields"]

    updated_fields = {"title": False, "description": False, "priority": False}

    full_match = re.match(
        r".*title\s*:\s*(.+?),\s*description\s*:\s*(.+?),\s*priority\s*:\s*(\w+)", user_input, re.IGNORECASE
    )
    if full_match:
        title, description, priority = full_match.groups()

        if not is_title_too_vague(title.strip()):
            fields["title"] = title.strip()
            updated_fields["title"] = True
            print(f"✔ Title set to: {fields['title']}")
        else:
            print("Title too vague — asking user for a clearer title.")

        fields["description"] = description.strip()
        updated_fields["description"] = True
        print("✔ Description set.")

        if priority.lower() in ["low", "medium", "high"]:
            fields["priority"] = priority.capitalize()
            updated_fields["priority"] = True
            print(f"✔ Priority set to: {fields['priority']}")
        return all(fields.values())

    title_match = re.search(r"title\s*:\s*([^,]+)", user_input, re.IGNORECASE)
    if title_match:
        title = title_match.group(1).strip()
        if not is_title_too_vague(title):
            fields["title"] = title
            updated_fields["title"] = True
            print(f"✔ Title set to: {fields['title']}")
        else:
            print("Title too vague — asking user for a clearer title.")

    desc_match = re.search(r"description\s*:\s*([^,]+)", user_input, re.IGNORECASE)
    if desc_match:
        fields["description"] = desc_match.group(1).strip()
        updated_fields["description"] = True
        print("✔ Description set.")

    priority_match = re.search(r"priority\s*:\s*(\w+)", user_input, re.IGNORECASE)
    if priority_match:
        priority = priority_match.group(1).strip().lower()
        if priority in ["low", "medium", "high"]:
            fields["priority"] = priority.capitalize()
            updated_fields["priority"] = True
            print(f"✔ Priority set to: {fields['priority']}")

    if not fields["priority"]:
        match = re.search(r"\b(low|medium|high)\b", lowered)
        if match:
            fields["priority"] = match.group(1).capitalize()
            updated_fields["priority"] = True
            print(f"✔ Priority set to: {fields['priority']}")

    if not fields["title"] and len(user_input.split()) <= 10 and not re.search(r"title\s*:", user_input, re.IGNORECASE):
        if not is_title_too_vague(user_input.strip()):
            fields["title"] = user_input.strip()
            updated_fields["title"] = True
            print(f"✔ Title set to: {fields['title']}")
        else:
            print("Title too vague — asking user for a clearer title.")

    if (
        fields["title"]
        and not fields["description"]
        and not desc_match
        and len(user_input.strip().split()) <= 12
        and not re.search(r"title\s*:|priority\s*:|description\s*:", user_input, re.IGNORECASE)
        and is_valid_description(user_input)
        and user_input.strip() != fields["title"]
    ):
        fields["description"] = user_input.strip()
        updated_fields["description"] = True
        print("✔ Description set (fallback).")
    elif fields["title"] and not fields["description"]:
        print("Skipping fallback: input too short, vague, or same as title.")

    print(f"📌 After extraction: {fields}")
    return all(fields.values())

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
    response = call_mistral(prompt).strip().lower()
    
    if "yes" in response:
        return True
    if "no" in response:
        return False
    return False

def is_title_too_vague(text):
    prompt = (
        "You are checking if a sentence from a user is suitable as an issue title in a bug tracking system.\n"
        "The title should be specific and describe the problem clearly.\n"
        "Reply only with 'Yes' if it's vague, intent-only, or generic.\n"
        "Reply with 'No' if the title describes a real problem clearly.\n\n"
        "Examples:\n"
        "- 'Jira dashboard is blank' → No\n"
        "- 'I want to create a ticket' → Yes\n"
        "- 'Server CPU reaches 100% on idle' → No\n"
        "- 'I have a problem' → Yes\n"
        "- 'Login page throws 500 error' → No\n\n"
        f"User input: \"{text.strip()}\"\n"
        "Is this too vague or intent-only?"
    )
    try:
        response = call_mistral(prompt).strip().lower()
        return "yes" in response
    except Exception as e:
        print("Failed vague check via Mistral:", e)
        return False

def is_valid_description(text):
    return len(text.split()) > 4 or text.endswith((".", "!", "?"))

def check_field_update_intent(user_input):
    prompt = f"""
You are a smart intent classifier. Based on the user's message, determine if they are trying to update one of the following fields of a ticket: title, description, or priority.

Reply only in one of these formats:
- "Update: title"
- "Update: description"
- "Update: priority"
- "None"

Message: "{user_input.strip()}"
What field is the user trying to update?
"""
    result = call_mistral(prompt).strip().lower()
    if "update: title" in result:
        return "title"
    elif "update: description" in result:
        return "description"
    elif "update: priority" in result:
        return "priority"
    return None