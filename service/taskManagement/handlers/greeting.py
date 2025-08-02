import re
from utils.llm import call_mistral

def is_greeting_only(user_input: str, state: dict = None) -> bool:
    """
    Use LLM to intelligently detect if user input is ONLY a greeting.
    Now includes context awareness to avoid false positives during specific flows.
    """
    # GUARD: Skip greeting detection during specific awaiting states
    if state:
        context_states = [
            "awaiting_title",
            "awaiting_description", 
            "awaiting_priority",
            "awaiting_ticket_confirmation",
            "awaiting_attachment_confirmation",
            "awaiting_file_upload",
            "awaiting_field_update",
            "awaiting_initial_choice"
        ]
        
        # If any of these states are active, don't treat input as greeting
        for context_state in context_states:
            if state.get(context_state):
                return False
        
        # Additional check for ticket flow states
        if state.get("ticket_flow_started") or state.get("kb_flow_started"):
            # Check if input matches priority options
            cleaned_input = user_input.lower().strip()
            priority_options = ["low", "medium", "high"]
            if cleaned_input in priority_options:
                return False
            
            # Check if input matches common button values
            button_values = ["yes", "no", "create", "task_management", "knowledge_base"]
            if cleaned_input in button_values:
                return False
    
    # Quick length check for very long inputs (likely not just greetings)
    if len(user_input.strip()) > 100:
        return False
    
    # Use Mistral to classify the input
    classification_prompt = f"""
Analyze this user input: "{user_input.strip()}"

Determine if this is ONLY a greeting/social pleasantry with no other intent, or if it contains requests for help, technical issues, or other business content.

ONLY a greeting examples:
- "Hi"
- "Hello there"
- "Good morning"
- "How are you?"
- "What's up?"
- "Hey buddy"
- "How's it going?"

NOT only greetings (contains other content):
- "Hi, I need help with my computer"
- "Hello, can you create a ticket?"
- "Good morning, I have an issue"
- "Hey, my system is not working"
- "Hi there, I want to search the knowledge base"
- Single words like "medium", "low", "high" (these are likely selections, not greetings)
- "yes", "no", "create" (these are responses/selections)

Respond with exactly one word:
- GREETING_ONLY (if it's just a greeting/social pleasantry)
- HAS_CONTENT (if it contains requests, issues, or other business content)
"""

    try:
        result = call_mistral(classification_prompt).strip().upper()
        
        # Parse the result
        if "GREETING_ONLY" in result:
            return True
        elif "HAS_CONTENT" in result:
            return False
        else:
            # If unclear response, use fallback logic
            return _fallback_greeting_check(user_input)
            
    except Exception as e:
        print(f"LLM classification failed: {e}")
        # Fallback to simple heuristics if LLM fails
        return _fallback_greeting_check(user_input)


def _fallback_greeting_check(user_input: str) -> bool:
    """
    Fallback method if LLM fails - uses minimal heuristics instead of complex regex
    """
    cleaned = user_input.lower().strip()
    
    # IMPORTANT: Exclude common button/selection values
    button_values = ["low", "medium", "high", "yes", "no", "create", "task_management", "knowledge_base"]
    if cleaned in button_values:
        return False
    
    # Very short common greetings
    simple_greetings = [
        'hi', 'hello', 'hey', 'yo', 'sup', 'morning', 'evening', 
        'afternoon', 'howdy', 'greetings', 'hiya', 'heya'
    ]
    
    if cleaned in simple_greetings:
        return True
    
    # Check for obvious business/technical keywords
    business_keywords = [
        'help', 'issue', 'problem', 'error', 'ticket', 'create', 
        'search', 'knowledge', 'broken', 'fix', 'support'
    ]
    
    for keyword in business_keywords:
        if keyword in cleaned:
            return False
    
    # If short and no business keywords, likely a greeting
    return len(cleaned) <= 20


def handle_greeting(user_input: str, state: dict, conversation: list) -> bool:
    """
    Handle greeting messages using LLM-based detection with context awareness.
    Returns True if a greeting was handled, False otherwise.
    """
    # Pass state to is_greeting_only for context awareness
    if is_greeting_only(user_input, state):
        state["greeted"] = True
        
        # Generate contextual greeting response
        greeting_response_prompt = f"""
The user sent this greeting: "{user_input.strip()}"

Generate a brief, friendly response that:
1. Acknowledges their greeting naturally
2. Offers help in one sentence
3. Sounds conversational, not robotic
4. Is no longer than 15 words

Examples:
- User: "Hi" → "Hello! How can I help you today?"
- User: "Good morning" → "Good morning! What can I assist you with?"
- User: "How are you?" → "I'm doing well, thanks! How can I help you?"
- User: "What's up?" → "Not much! What can I do for you?"

Respond with just the greeting response, nothing else.
"""
        
        try:
            response = call_mistral(greeting_response_prompt).strip()
            
            # Validate response quality
            if len(response) > 100 or not response:
                response = "Hello! How can I help you today?"
                
            conversation.append({"role": "assistant", "content": response})
            print(f"Starlistant: {response}")
            return True
            
        except Exception as e:
            print(f"Greeting response generation failed: {e}")
            # Simple fallback
            fallback_response = "Hello! How can I help you today?"
            conversation.append({"role": "assistant", "content": fallback_response})
            print(f"Starlistant: {fallback_response}")
            return True
    
    return False


def contains_greeting_and_content(user_input: str) -> bool:
    """
    Use LLM to detect if input contains both greeting AND business content.
    """
    if len(user_input.strip()) <= 15:  # Too short to have both
        return False
    
    mixed_content_prompt = f"""
Analyze this user input: "{user_input.strip()}"

Does this contain BOTH:
1. A greeting/pleasantry (hi, hello, good morning, etc.)
2. AND business content (requests for help, technical issues, ticket creation, etc.)

Examples of MIXED content:
- "Hi, I need help with my computer"
- "Good morning, can you create a ticket?"
- "Hello there, I have an issue with login"

Examples of NOT mixed:
- "Hi" (greeting only)
- "I need help" (business only)  
- "Good morning" (greeting only)

Respond with exactly:
- MIXED (if it contains both greeting and business content)
- NOT_MIXED (if it's only greeting or only business content)
"""

    try:
        result = call_mistral(mixed_content_prompt).strip().upper()
        return "MIXED" in result
    except Exception as e:
        print(f"Mixed content detection failed: {e}")
        # Simple fallback: if it's long and starts with common greetings
        cleaned = user_input.lower().strip()
        greeting_words = ['hi', 'hello', 'hey', 'good morning', 'good evening']
        return any(cleaned.startswith(word) for word in greeting_words) and len(cleaned) > 20