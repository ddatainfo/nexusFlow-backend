import re
from utils.llm import call_mistral

def is_greeting_only(user_input: str, state: dict = None) -> bool:
    """
    Use Mistral LLM to intelligently detect if user input is ONLY a greeting.
    Context-aware to avoid false positives during specific flows.
    No hardcoded values - relies entirely on LLM intelligence.
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
        
        # Additional check for active flows - let Mistral handle content validation
        if state.get("ticket_flow_started") or state.get("kb_flow_started"):
            # Use Mistral to check if this might be flow-related input
            context_check_prompt = f"""
User input: "{user_input.strip()}"

Context: The user is currently in a ticket creation or knowledge base flow.

Is this input likely to be:
A) A greeting/social pleasantry (like "hi", "hello", "how are you")
B) Flow-related content (like priority selections, yes/no answers, technical requests, information queries, names, topics)

Even if the input seems like a greeting, if we're in an active flow, it might be intended as content.

Examples in flow context:
- "medium" = flow content (priority selection)
- "yes" = flow content (confirmation)  
- "Dr. Smith" = flow content (name/topic)
- "BERT expansion" = flow content (technical topic)
- "Who is..." = flow content (information query)
- "hi" = could be greeting even in flow
- "hello there" = could be greeting even in flow

Respond with exactly one word:
- GREETING (if it's clearly just a social greeting)
- CONTENT (if it's likely flow-related content or information request)
"""
            
            try:
                context_result = call_mistral(context_check_prompt).strip().upper()
                if "CONTENT" in context_result:
                    return False
                # If GREETING, continue with normal greeting detection
            except Exception as e:
                print(f"Context check failed: {e}")
                # If context check fails, err on side of caution and don't treat as greeting
                return False
    
    # Use Mistral to classify the input with comprehensive prompt
    classification_prompt = f"""
Analyze this user input: "{user_input.strip()}"

Your task: Determine if this is ONLY a greeting/social pleasantry with NO other meaningful content, requests, or information-seeking intent.

GREETING_ONLY examples:
- "Hi"
- "Hello" 
- "Hey"
- "Good morning"
- "How are you?"
- "What's up?" (when used casually as greeting)
- "How's it going?"
- "Greetings"
- "Hey there"
- "Morning"

HAS_CONTENT examples (NOT just greetings):
- "Who is Dr. V. Karpagam" (information request about a person)
- "What is SKLEARN" (information request about a topic)
- "Expansion of BERT" (request for information on a topic)
- "Machine learning help" (request for assistance)
- "Python tutorial" (request for educational content)
- "Tell me about..." (explicit information request)
- "Explain..." (request for explanation)
- "How to..." (request for instructions)
- "Create a ticket" (action request)
- "Search knowledge base" (action request)
- Any questions starting with: who, what, where, when, why, how (when asking for information)
- Any names, people, places, technologies, concepts being mentioned
- Any technical terms, academic topics, professional subjects
- Any requests for help, information, explanations, tutorials
- Single word responses like "yes", "no", "create", "medium", "high" (these are selections/responses)
- Numbers, selections, confirmations

CRITICAL ANALYSIS POINTS:
1. Questions seeking information (who/what/where/when/why/how + content) = HAS_CONTENT
2. Any mention of specific people, places, topics, technologies = HAS_CONTENT  
3. Any requests for help, explanation, information = HAS_CONTENT
4. Only pure social interactions with no information-seeking intent = GREETING_ONLY
5. When in doubt about whether it's seeking information, classify as HAS_CONTENT

Think step by step:
1. Is this asking for any information about anything or anyone?
2. Is this mentioning any specific topics, names, or subjects?
3. Is this requesting any kind of help or assistance?
4. Is this trying to accomplish any task or get any information?

If YES to any of the above → HAS_CONTENT
If NO to all (pure social greeting only) → GREETING_ONLY

Respond with exactly one word:
- GREETING_ONLY 
- HAS_CONTENT
"""

    try:
        result = call_mistral(classification_prompt).strip().upper()
        
        # Parse the result
        if "GREETING_ONLY" in result:
            return True
        elif "HAS_CONTENT" in result:
            return False
        else:
            # If unclear response, use basic Mistral-based fallback
            return _mistral_fallback_check(user_input)
            
    except Exception as e:
        print(f"LLM classification failed: {e}")
        # Fallback to another Mistral call with simpler prompt
        return _mistral_fallback_check(user_input)


def _mistral_fallback_check(user_input: str) -> bool:
    """
    Fallback method using Mistral with simpler prompt if main classification fails
    """
    simple_prompt = f"""
Is this ONLY a greeting: "{user_input.strip()}"

A greeting is purely social with no information requests.
Examples: "Hi", "Hello", "How are you?"

Not greetings: questions, requests, topics, names, technical terms.

Answer: YES (greeting only) or NO (has other content)
"""
    
    try:
        result = call_mistral(simple_prompt).strip().upper()
        return "YES" in result
    except Exception as e:
        print(f"Fallback classification failed: {e}")
        # Ultimate fallback - be conservative and assume it has content
        return False


def handle_greeting(user_input: str, state: dict, conversation: list) -> bool:
    """
    Handle greeting messages using pure LLM-based detection.
    Returns True if a greeting was handled, False otherwise.
    """
    # Use Mistral to determine if this is a greeting
    if is_greeting_only(user_input, state):
        state["greeted"] = True
        
        # Generate contextual greeting response using Mistral
        greeting_response_prompt = f"""
The user sent this greeting: "{user_input.strip()}"

Generate a brief, friendly response that:
1. Acknowledges their greeting naturally
2. Offers help in a conversational way
3. Sounds warm and human, not robotic
4. Is concise (10-20 words max)
5. Ends with offering assistance

Examples:
- User: "Hi" → "Hello! How can I help you today?"
- User: "Good morning" → "Good morning! What can I assist you with?"
- User: "How are you?" → "I'm doing well, thanks! How can I help you?"
- User: "What's up?" → "Not much! What can I do for you today?"
- User: "Hey there" → "Hey! What brings you here today?"

Generate only the response text, nothing else.
"""
        
        try:
            response = call_mistral(greeting_response_prompt).strip()
            
            # Validate response quality using Mistral
            if len(response) > 150 or len(response) < 5:
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
    Use Mistral to detect if input contains both greeting AND meaningful content.
    """
    mixed_content_prompt = f"""
Analyze this user input: "{user_input.strip()}"

Does this contain BOTH:
1. A greeting/social pleasantry (hi, hello, good morning, etc.)
AND
2. Meaningful content (requests for help, information, topics, technical content, etc.)

Examples of MIXED content (has both greeting and meaningful content):
- "Hi, I need help with my computer"
- "Good morning, can you create a ticket?"
- "Hello there, I have an issue with login"
- "Hey, tell me about machine learning"
- "Hi, who is Dr. Smith?"

Examples of NOT mixed (only one or the other):
- "Hi" (greeting only)
- "Hello" (greeting only)
- "I need help" (content only)  
- "Good morning" (greeting only)
- "Who is Dr. Smith?" (content only)
- "Expansion of SKLEARN" (content only)

Analyze the input carefully:
1. Does it start with or contain social greetings?
2. Does it also ask for information, help, or mention specific topics?

Respond with exactly one word:
- MIXED (contains both greeting and meaningful content)
- NOT_MIXED (contains only greeting OR only content, but not both)
"""

    try:
        result = call_mistral(mixed_content_prompt).strip().upper()
        return "MIXED" in result
    except Exception as e:
        print(f"Mixed content detection failed: {e}")
        # Conservative fallback - assume not mixed
        return False