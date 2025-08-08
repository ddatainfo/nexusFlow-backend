import json
import re
from datetime import datetime, timedelta
import requests
from typing import Dict
from dateutil.parser import parse as parse_date
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

def extract_json_from_response(response: str, user_input: str) -> Dict:
    """
    Extracts the first valid { ... } block from response as JSON.
    Falls back to inferring parameters from response or input if JSON is invalid.
    """
    if not response:
        logger.error("LLM response is empty, inferring from input")
        return infer_params_from_input(user_input)

    # Remove ```json, ```, or any narrative text before/after JSON
    cleaned_response = re.sub(r'```json\s*|\s*```|.*?(\{[\s\S]*?\}).*', r'\1', response, flags=re.DOTALL).strip()
    try:
        return json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        logger.error(f"JSON extraction failed: {e}")
        logger.warning("No valid JSON found, inferring parameters")
        return infer_params_from_input(user_input)

def infer_params_from_input(user_input: str) -> Dict[str, str]:
    """
    Fallback to infer parameters from user input using regex and dateutil.
    """
    username = ""
    status = ""
    from_date = ""
    to_date = ""

    # Extract username (skip common words)
    username_match = re.search(r"\b(?!this|last|week|task|tasks|for\b)[A-Za-z][A-Za-z0-9]*\b", user_input, re.IGNORECASE)
    if username_match:
        username = username_match.group(0)

    # Extract status
    status_pattern = r'\b(?:in progress|done|completed|pending|open|to do)\b'
    status_match = re.search(status_pattern, user_input, re.IGNORECASE)
    if status_match:
        status = status_match.group(0).lower()
        if status == "pending":
            status = "to do"  # Map 'pending' to Jira's 'To Do'

    # Extract dates using dateutil
    date_pattern = r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}|today|yesterday|this week|last week)\b'
    dates = re.findall(date_pattern, user_input, re.IGNORECASE)
    if dates:
        try:
            parsed_date = parse_date(dates[0], dayfirst=True)
            from_date = parsed_date.strftime("%Y-%m-%d")
            to_date = from_date if len(dates) == 1 else parse_date(dates[1], dayfirst=True).strftime("%Y-%m-%d")
        except ValueError:
            logger.warning(f"Failed to parse dates in input: {user_input}")

    return {"username": username, "status": status, "from_date": from_date, "to_date": to_date}

def resolve_special_dates(from_date_str: str, to_date_str: str) -> tuple[str, str]:
    """
    Normalize special date terms and parsed dates to YYYY-MM-DD format.
    """
    today = datetime.utcnow().date()
    from_lower = (from_date_str or "").lower()
    to_lower = (to_date_str or "").lower()

    if from_lower in ["today"] or to_lower in ["today"]:
        day_str = today.strftime("%Y-%m-%d")
        return day_str, day_str
    if from_lower in ["yesterday"] or to_lower in ["yesterday"]:
        yesterday = today - timedelta(days=1)
        day_str = yesterday.strftime("%Y-%m-%d")
        return day_str, day_str
    if from_lower in ["this week", "last week"] or to_lower in ["this week", "last week"]:
        week_start = (today - timedelta(days=7)).strftime("%Y-%m-%d")
        week_end = today.strftime("%Y-%m-%d")
        return week_start, week_end

    # Parse dates with dateutil
    try:
        if from_date_str:
            from_date = parse_date(from_date_str, dayfirst=True).strftime("%Y-%m-%d")
        else:
            from_date = ""
        if to_date_str:
            to_date = parse_date(to_date_str, dayfirst=True).strftime("%Y-%m-%d")
        else:
            to_date = from_date if from_date else ""
        return from_date, to_date
    except ValueError:
        logger.warning(f"Invalid date format: from_date={from_date_str}, to_date={to_date_str}")
        return "", ""

def parse_jira_query_with_mistral(user_input: str) -> Dict[str, str]:
    logger.debug(f"Parsing user input with LLM: {user_input}")
    prompt = f"""
    Return only a JSON object with the following fields extracted from the Jira query. Do not include any additional text, comments, code block markers (like ```json or ```), or explanations. Ensure the response is a single, valid JSON object.

    Fields:
    - username: The proper noun or alphanumeric string at the start of the query, before 'task'/'tasks', or after prepositions like 'for' (e.g., in 'task for User'), else empty string
    - status: Task status (e.g., 'in progress', 'done', 'completed', 'pending', 'open', 'to do') if explicitly mentioned, map 'pending' to 'to do', else empty string
    - from_date: Start date in YYYY-MM-DD format; for relative terms like 'today' use {datetime.now().strftime('%Y-%m-%d')}, 'yesterday' use {(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')}, 'this week' or 'last week' use {(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')}, or parse dates like 'DD-MM-YY', 'DD/MM/YYYY', 'Month DD, YYYY' (e.g., 'July 1st, 2025'), else empty string
    - to_date: End date in YYYY-MM-DD format; for relative terms or parsed dates same as from_date, else empty string

    Query: "{user_input}"

    Example output:
    {{
        "username": "",
        "status": "",
        "from_date": "",
        "to_date": ""
    }}
    """
    logger.debug(f"Calling Mistral with prompt: {prompt[:100]}...")
    response = call_mistral(prompt)
    logger.debug(f"Raw LLM response: {response}")
    try:
        parsed = json.loads(response)
    except Exception as e:
        logger.error(f"Failed to parse LLM response: {e}")
        parsed = extract_json_from_response(response, user_input)

    # Fallback: Ensure username is extracted if LLM fails
    if not parsed.get("username"):
        username_match = re.search(r"\b(?!this|last|week|task|tasks|for\b)[A-Za-z][A-Za-z0-9]*\b", user_input, re.IGNORECASE)
        if username_match:
            parsed["username"] = username_match.group(0)
            logger.debug(f"Fallback extracted username: {parsed['username']}")

    # Normalize username
    if parsed.get("username"):
        parsed["username"] = " ".join(parsed["username"].split()).split()[0]

    # Map 'pending' to 'to do'
    if parsed.get("status", "").lower() == "pending":
        parsed["status"] = "to do"

    # Validate status
    valid_statuses = {"done", "completed", "pending", "open", "in progress", "to do"}
    if parsed.get("status", "").lower() not in valid_statuses:
        parsed["status"] = ""

    # Normalize and validate dates with dateutil
    from_date, to_date = resolve_special_dates(parsed.get("from_date"), parsed.get("to_date"))
    parsed["from_date"] = from_date
    parsed["to_date"] = to_date

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

    return parsed