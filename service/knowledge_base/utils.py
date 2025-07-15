import re
import ollama
from chromadb import PersistentClient
from embedding_model import get_embedder
from pathlib import Path
from fastapi import HTTPException

WORK_ROOT = Path("/home/ddata/work/nantha/chatbot_api/pdf_service/rag_workspace")
CHROMA_DB_DIR = str(WORK_ROOT / "dev_chromadb")
COLLECTION_NAME = "dev_embeddings"
SIMILARITY_THRESHOLD = 0.35

def is_greeting(text):
    """Check if the user input is a greeting."""
    return bool(re.search(r"\b(hi+|hello+|hey+|good (morning|afternoon|evening))\b", text.lower()))

def build_prompt(query: str, context: dict, beautified_text, chat_history=None) -> str:
    """
    Builds a structured and constrained prompt for the LLM to avoid hallucination, including last 5 chat history.
    """
    prompt = (
        "You are a highly accurate assistant. "
        "Only answer questions using the information provided in the context below. "
        "**If you cannot find the answer in the context, say: 'The context is out of my knowledge.'** "
        "Do not guess or assume anything outside of the context.\n\n"
    )
    # Add chat history if available
    if chat_history:
        prompt += "=== CHAT HISTORY (last 5) ===\n"
        for turn in chat_history:
            prompt += f"User: {turn['question']}\nAssistant: {turn['answer']}\n"
        prompt += "\n"
    # Add beautified text
    if beautified_text:
        prompt += f"=== BEAUTIFIED TEXT ===\n{beautified_text}\n\n"
    # Add table summaries
    if context.get("table"):
        prompt += "=== TABLES ===\n"
        for table, meta in context["table"]:
            page = meta.get("page", "unknown")
            prompt += f"- Page {page}: {table.strip()}\n"
        prompt += "\n"
    # Add image captions
    if context.get("image"):
        prompt += "=== IMAGES ===\n"
        for image_caption, meta in context["image"]:
            prompt += f"- {image_caption} (image_id: {meta.get('id')})\n"
        prompt += "\n"
    # Final instruction with beautified query
    prompt += (
        f"=== QUESTION ===\n{query}\n\n"
        "=== ANSWER ==="
    )
    return prompt

def validator(beautify_data, query):
    prompt = (
        "You are an validation specialist. "
        "where you understand the given data and validate it and you say it can able to answer the query given by the user."
        f"\n\nContext to verify :\n{beautify_data}\n\n"
        f"\n\n query :\n{query}\n\n"
        "just say 'yes' if it can answer the query, otherwise say 'no'. "
        ""
    )
    response = ollama.chat(
            model="mistral",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Respond naturally."},
                {"role": "user", "content": prompt}
            ],
            options={"temperature": 0.4}
        )
    response_text = response["message"]["content"].strip()
    print("--------------------------------")
    print("validator response:", response_text)
    print("--------------------------------")
    return response_text

def beautify_text(data):
    prompt = (
        "you are a inspection specialist. "
        " where you inspect the given data and format it in a nice way to pass the data to next step."
        " which helps in order to generate a answer from it."
        " make sure dont change any context , just format and beautify."
        f"\n\nInput text:\n{data}\n\n"
    )
    response = ollama.chat(
            model="mistral",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Respond naturally."},
                {"role": "user", "content": prompt}
            ],
            options={"temperature": 0.4}
        )
    response_text = response["message"]["content"].strip()
    print("--------------------------------")
    print("beautify response:", response_text)
    print("--------------------------------")
    return response_text

def chat_with_knowledge(query, id):
    print("Query received", query)
    if is_greeting(query):
        return {"response": "Hi there! How can I assist you today?"}
    client = PersistentClient(path=CHROMA_DB_DIR)
    coll = client.get_collection(COLLECTION_NAME)
    # Embed query
    query_vec = get_embedder().encode([query])[0]
    # Retrieve from ChromaDB
    res = coll.query(
        query_embeddings=[query_vec],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )
    print("Ask endpoint response:", res['documents'])

    if not res["documents"] or not res["documents"][0]:
        raise HTTPException(404, "No context found")
    # Filter by similarity
    beautify_data = beautify_text(res["documents"][0])
    print("Beautified data:", beautify_data)
    validate_response = validator(beautify_data, query)
    if "yes" in validate_response.lower():
        print("i can generate answer")
        grouped = {"text": [], "table": [], "image": []}

        filtered = []
        for i, doc in enumerate(res["documents"][0]):
            dist = res["distances"][0][i]
            print("Distance for doc", i, ":", dist)
            filtered.append({
                "doc": doc,
                "meta": res["metadatas"][0][i],
            })

        for item in filtered:
            doc = item["doc"]
            meta = item["meta"]
            t = meta.get("type", "text")
            grouped[t].append((doc, meta))
        # Get last 5 chat history for this conversation
        from conversation_state import conversation_states
        state = conversation_states.get(id, {})
        chat_history = state.get("chat_history", [])
        # Build prompt with chat history
        prompt = build_prompt(query, grouped, beautify_data, chat_history)
        # Call LLM
        try:
            print("Prompt for Mistral:")
            response = ollama.chat(
                model="mistral",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Respond naturally."},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.4}
            )
            response_text = response["message"]["content"].strip()
            print("Mistral response:", response_text)
            # Update chat history
            from conversation_state import update_chat_history
            update_chat_history(id, query, response_text)
            return {"response": response_text}
        except Exception as e:
            raise HTTPException(500, str(e))
    elif "no" in validate_response.lower():
        prompt = (
            "It's seems like it's out of my knowledge or i can't get the details clearly "
            " I can't answer! "
            "Please ask something about the knowledge which is updated."
        )
        return {"response": prompt}