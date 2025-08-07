import json
import re
from datetime import datetime, timedelta
import requests
from app.config.settings import OLLAMA_BASE_URL, MODEL_NAME
import logging
import time

logger = logging.getLogger(__name__)

def call_mistral(prompt: str, retries: int = 3, timeout: int = 60) -> str:
    """
    Calls the local Mistral LLM via Ollama API with retry logic.
    Returns the raw response (expected as a JSON string).
    """
    for attempt in range(retries):
        try:
            res = requests.post(
                OLLAMA_BASE_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3,
                    "top_p": 0.8,
                    "repeat_penalty": 1.2,
                },
                timeout=timeout,
            )
            res.raise_for_status()
            response = res.json().get("response", "").strip()
            logger.debug(f"Mistral raw response: {response}")
            return response
        except Exception as e:
            logger.error(f"❌ Mistral call failed (attempt {attempt + 1}/{retries}): {e}")
            if attempt == retries - 1:
                logger.error("All retries failed, returning empty string")
                return ""
            time.sleep(2)  # Wait before retrying
    return ""

def extract_json_from_response(response: str, user_input: str) -> dict:
    """
    Extracts the first valid { ... } block from response as JSON.
    Falls back to inferring parameters from response or input if JSON is invalid.
    """
    if not response:
        logger.error("LLM response is empty, inferring from input")
        return infer_params_from_input(user_input)

    # Try to find a JSON block using regex
    match = re.search(r'\{[\s\S]*?\}', response)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as e:
            logger.error(f"JSON extraction failed: {e}")

    # Fallback: infer parameters from response or input
    logger.warning("No valid JSON found, inferring parameters")
    return infer_params_from_input(user_input)

def infer_params_from_input(user_input: str) -> dict:
    """
    Infers username, status, and dates from user input when LLM response is invalid.
    """
    parsed = {"username": "", "status": "", "from_date": "", "to_date": ""}
    today = datetime.utcnow().date()
    yesterday = today - timedelta(days=1)

    # Infer username (any alphanumeric string, optionally followed by space and initial)
    username_match = re.search(r'\b([a-zA-Z0-9]+)(?:\s+[a-zA-Z])?\b', user_input, re.IGNORECASE)
    if username_match:
        parsed["username"] = username_match.group(1)

    # Infer status
    valid_statuses = {"done", "completed", "pending", "open", "in progress", "to do"}
    for status in valid_statuses:
        if status.lower() in user_input.lower():
            parsed["status"] = status
            break

    # Infer dates
    if "today" in user_input.lower():
        parsed["from_date"] = today.strftime("%Y-%m-%d")
        parsed["to_date"] = today.strftime("%Y-%m-%d")
    elif "yesterday" in user_input.lower():
        parsed["from_date"] = yesterday.strftime("%Y-%m-%d")
        parsed["to_date"] = yesterday.strftime("%Y-%m-%d")
    else:
        # Look for date range (e.g., "from 2025-07-01 to 2025-08-07")
        range_match = re.search(r'from\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})', user_input, re.IGNORECASE)
        if range_match:
            parsed["from_date"] = range_match.group(1)
            parsed["to_date"] = range_match.group(2)
        else:
            # Look for single date (e.g., "2025-08-06" or "0205-08-06")
            date_match = re.search(r'\b(\d{4}|\d{2,4}-\d{2}-\d{2})\b', user_input)
            if date_match:
                date_str = date_match.group(0)
                if date_str.startswith("0") and len(date_str.split("-")[0]) == 4:
                    date_str = "2" + date_str[1:]
                parsed["from_date"] = date_str
                parsed["to_date"] = date_str

    return parsed

def resolve_special_dates(from_date_str: str, to_date_str: str) -> tuple[str, str]:
    """Normalize 'today', 'yesterday' to actual dates, handle missing one date as single day."""
    today = datetime.utcnow().date()
    from_lower = (from_date_str or "").lower()
    to_lower = (to_date_str or "").lower()

    if from_lower == "today" or to_lower == "today":
        day_str = today.strftime("%Y-%m-%d")
        return day_str, day_str
    if from_lower == "yesterday" or to_lower == "yesterday":
        yesterday = today - timedelta(days=1)
        day_str = yesterday.strftime("%Y-%m-%d")
        return day_str, day_str
    if from_date_str and not to_date_str:
        return from_date_str, from_date_str
    if to_date_str and not from_date_str:
        return to_date_str, to_date_str
    return from_date_str, to_date_str

def parse_jira_query_with_mistral(user_input: str) -> dict:
    today = datetime.utcnow().date()
    yesterday = today - timedelta(days=1)
    prompt = f'''
Extract the following fields from this Jira query in natural English: username, status, from_date, to_date.

User query: "{user_input}"

Rules:
- Username: Extract the username as a single string, removing extra spaces or initials (e.g., "DhanushKanna G" becomes "DhanushKanna"). If no username is specified, return an empty string.
- Status: Only extract a status if it is explicitly one of: "done", "completed", "pending", "open", "in progress", "to do". Ignore generic terms like "task", "tasks", or "issues". If no valid status is specified, return an empty string.
- Dates:
  - If "today" is mentioned, set both from_date and to_date to "{today.strftime('%Y-%m-%d')}".
  - If "yesterday" is mentioned, set both from_date and to_date to "{yesterday.strftime('%Y-%m-%d')}".
  - If a specific date is mentioned (e.g., "2025-08-06" or "on 2025-08-06"), set both from_date and to_date to that date unless a range is explicitly provided.
  - If a date range is mentioned with "from ... to ..." (e.g., "from 2025-07-01 to 2025-08-07"), set from_date to the first date and to_date to the second date.
  - For dates with ambiguous years (e.g., "0205-08-06"), interpret as "2025" (assume years in 2000–2099).
  - Validate dates to ensure they are in "YYYY-MM-DD" format and plausible (between 2010 and 2030). If invalid, set both dates to empty strings.
  - If no dates are specified, return empty strings for both from_date and to_date.
- Output: Return ONLY a single valid JSON object with the fields username, status, from_date, and to_date. Do NOT include explanations, code blocks, Markdown, or any text outside the JSON object. Non-compliance will result in parsing errors.

Example inputs and expected outputs:
- Input: "DhanushKanna G tasks today"
  Output: {{"username": "DhanushKanna", "status": "", "from_date": "{today.strftime('%Y-%m-%d')}", "to_date": "{today.strftime('%Y-%m-%d')}"}}
- Input: "DhanushKanna G pending task on 2025-08-06"
  Output: {{"username": "DhanushKanna", "status": "pending", "from_date": "2025-08-06", "to_date": "2025-08-06"}}
- Input: "DhanushKanna G tasks from 2025-07-01 to 2025-08-07"
  Output: {{"username": "DhanushKanna", "status": "", "from_date": "2025-07-01", "to_date": "2025-08-07"}}
- Input: "DhanushKanna G task 0205-08-06 In progress"
  Output: {{"username": "DhanushKanna", "status": "in progress", "from_date": "2025-08-06", "to_date": "2025-08-06"}}

Output format:
{{
  "username": "",
  "status": "",
  "from_date": "",
  "to_date": ""
}}
'''
    logger.debug(f"Parsing user input with LLM: {user_input}")
    response = call_mistral(prompt)
    logger.debug(f"Raw LLM response: {response}")
    try:
        parsed = json.loads(response)
    except Exception as e:
        logger.error(f"Failed to parse LLM response: {e}")
        parsed = extract_json_from_response(response, user_input)

    # Normalize username
    if parsed.get("username"):
        parsed["username"] = " ".join(parsed["username"].split()).split()[0]

    # Validate status
    valid_statuses = {"done", "completed", "pending", "open", "in progress", "to do"}
    if parsed.get("status", "").lower() not in valid_statuses:
        parsed["status"] = ""

    # Normalize and validate dates
    for date_field in ["from_date", "to_date"]:
        if parsed.get(date_field):
            date_str = parsed[date_field]
            if date_str.startswith("0") and len(date_str.split("-")[0]) == 4:
                date_str = "2" + date_str[1:]
            try:
                parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
                year = parsed_date.year
                if year < 2010 or year > 2030:
                    logger.warning(f"Date {date_str} outside valid range (2010–2030), setting to empty")
                    parsed["from_date"] = ""
                    parsed["to_date"] = ""
                    break
                parsed[date_field] = parsed_date.strftime("%Y-%m-%d")
            except ValueError:
                logger.warning(f"Invalid date format: {date_str}, setting to empty")
                parsed["from_date"] = ""
                parsed["to_date"] = ""
                break

    # Ensure from_date <= to_date
    if parsed.get("from_date") and parsed.get("to_date"):
        try:
            from_date = datetime.strptime(parsed["from_date"], "%Y-%m-%d")
            to_date = datetime.strptime(parsed["to_date"], "%Y-%m-%d")
            if from_date > to_date:
                logger.warning(f"from_date {parsed['from_date']} is after to_date {parsed['to_date']}, swapping")
                parsed["from_date"], parsed["to_date"] = parsed["to_date"], parsed["from_date"]
        except ValueError:
            parsed["from_date"] = ""
            parsed["to_date"] = ""
    elif parsed.get("from_date") and not parsed.get("to_date"):
        parsed["to_date"] = parsed["from_date"]
    elif parsed.get("to_date") and not parsed.get("from_date"):
        parsed["from_date"] = parsed["to_date"]

    return parsed