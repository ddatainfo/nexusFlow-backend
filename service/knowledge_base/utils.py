#===================UTILS WITH RERANKER===================

import re
import ollama
import numpy as np
from chromadb import PersistentClient
from embedding_model import get_embedder
from pathlib import Path
from fastapi import HTTPException
from sentence_transformers import CrossEncoder  # NEW: reranker
import re

WORK_ROOT = Path("/home/ddata/dev/nexusFlow-backend/service/rag_workspace")
CHROMA_DB_DIR = str(WORK_ROOT / "dev_chromadb")
COLLECTION_NAME = "dev_embeddings"

# Retrieval knobs
INITIAL_K = 20          # pull more for reranking
CONTEXT_K = 10           # how many to pass to the LLM
MAX_SNIPPET_CHARS = 1200  # safety clamp for beautify/LLM

# NEW: lightweight cross-encoder reranker
_reranker = None
def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("BAAI/bge-reranker-base")
    return _reranker

def _normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec) + 1e-12
    return vec / n

def _strip_source_tags(text: str) -> str:
    # remove [S1], [S2], … and any adjacent commas/spaces left behind
    text = re.sub(r'(\s*\[\s*S\d+\s*\]\s*,?)+', '', text)
    # collapse double spaces created by removals
    return re.sub(r'\s{2,}', ' ', text).strip()

def is_greeting(text):
    """Check if the user input is a greeting."""
    return bool(re.search(r"\b(hi+|hello+|hey+|good (morning|afternoon|evening))\b", text.lower()))

def build_prompt(query: str, context: dict, beautified_text: str, chat_history=None, sources=None) -> str:
    """
    Builds a structured and constrained prompt for the LLM to avoid hallucination, including last 5 chat history.
    Adds explicit source-citation instructions using [S#] tags.
    """
    prompt = (
        "You are a highly accurate assistant.\n"
        "Only answer using the information provided below. Do not guess.\n"
        "If the answer is not clearly supported, reply exactly: The context is out of my knowledge.\n"
        "When you present any numeric value or claim, cite the supporting snippet like [S1], [S2].\n\n"
    )
    # Add chat history if available
    if chat_history:
        prompt += "=== CHAT HISTORY (last 5) ===\n"
        for turn in chat_history:
            prompt += f"User: {turn['question']}\nAssistant: {turn['answer']}\n"
        prompt += "\n"

    # Add sources list with [S#] tags (these are short snippets or full chunks)
    if sources:
        prompt += "=== SOURCES ===\n"
        for i, s in enumerate(sources, start=1):
            # keep sources compact
            s = s.strip()
            if len(s) > 1000:
                s = s[:1000] + "…"
            prompt += f"[S{i}] {s}\n\n"

    # Add beautified aggregation (kept for your pipeline visibility)
    if beautified_text:
        prompt += f"=== BEAUTIFIED AGGREGATE ===\n{beautified_text}\n\n"

    # Add table summaries (if present)
    if context.get("table"):
        prompt += "=== TABLES ===\n"
        for table, meta in context["table"]:
            page = meta.get("page", "unknown")
            prompt += f"- Page {page}: {table.strip()}\n"
        prompt += "\n"

    # Add image captions (if present)
    if context.get("image"):
        prompt += "=== IMAGES ===\n"
        for image_caption, meta in context["image"]:
            prompt += f"- {image_caption} (image_id: {meta.get('id')})\n"
        prompt += "\n"

    # Final instruction with the question
    prompt += (
        f"=== QUESTION ===\n{query}\n\n"
        "=== ANSWER ==="
    )
    return prompt

def validator(beautified_aggregate: str, query: str) -> str:
    """
    Validate whether the combined (multi-snippet) aggregate likely contains enough signal.
    Much safer than validating only the first document.
    """
    prompt = (
        "You are a validation specialist. Determine if the following aggregated context contains enough information "
        "to answer the user's question. Reply with 'yes' or 'no' only.\n\n"
        f"Context:\n{beautified_aggregate}\n\n"
        f"Question:\n{query}\n\n"
        "Respond with 'yes' if the answer is clearly supported; otherwise 'no'."
    )
    response = ollama.chat(
        model="mistral",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Respond naturally."},
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0.2}
    )
    response_text = response["message"]["content"].strip()
    response_text = _strip_source_tags(response_text)   # <-- add this
    print("--------------------------------")
    print("validator response:", response_text)
    print("--------------------------------")
    return response_text

def beautify_text(data: str) -> str:
    """
    Keep your beautification step, but give it the combined snippets (tagged).
    """
    # Clamp to avoid overlong prompts
    if len(data) > MAX_SNIPPET_CHARS * 6:
        data = data[:MAX_SNIPPET_CHARS * 6] + "…"

    prompt = (
        "You are an inspection specialist. Format the given text cleanly without changing meaning.\n"
        "Do NOT add new claims. Preserve tables/bullets if present. Keep it concise.\n\n"
        f"Input text:\n{data}\n\n"
    )
    response = ollama.chat(
        model="mistral",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Respond naturally."},
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0.2}
    )
    response_text = response["message"]["content"].strip()
    print("--------------------------------")
    print("beautify response:", response_text)
    print("--------------------------------")
    return response_text

def chat_with_knowledge(query: str, id: str):
    print("Query received", query)
    if is_greeting(query):
        return {"response": "Hi there! How can I assist you today?"}

    # Connect to Chroma
    client = PersistentClient(path=CHROMA_DB_DIR)
    coll = client.get_collection(COLLECTION_NAME)

    # Embed + normalize query (consistency with cosine similarity)
    q = get_embedder().encode([query])[0]
    query_vec = _normalize(np.array(q))

    # Retrieve more, then rerank
    res = coll.query(
        query_embeddings=[query_vec],
        n_results=INITIAL_K,
        include=["documents", "metadatas", "distances"]
    )
    docs = res.get("documents", [[]])[0] or []
    metas = res.get("metadatas", [[]])[0] or []
    dists = res.get("distances", [[]])[0] or []
    print("Ask endpoint response (top-k raw):", docs)

    if not docs:
        raise HTTPException(404, "No context found")

    # Rerank using cross-encoder on (query, doc) pairs
    reranker = get_reranker()
    pairs = [(query, d) for d in docs]
    scores = reranker.predict(pairs)  # higher is better
    ranked = sorted(zip(docs, metas, dists, scores), key=lambda x: x[3], reverse=True)

    # Keep top CONTEXT_K
    top = ranked[:CONTEXT_K]
    top_docs = [t[0] for t in top]
    top_metas = [t[1] for t in top]
    top_scores = [t[3] for t in top]

    print("Reranked top scores:", top_scores)

    # Build grouped context by type
    grouped = {"text": [], "table": [], "image": []}
    for doc, meta in zip(top_docs, top_metas):
        t = meta.get("type", "text")
        grouped.setdefault(t, []).append((doc, meta))

    # Build source-tagged list for prompt + beautify combined text
    # Prefix each snippet with [S#] to enable hard citations
    sources = []
    combined_for_beautify = []
    for i, doc in enumerate(top_docs, start=1):
        snippet = doc.strip()
        if len(snippet) > MAX_SNIPPET_CHARS:
            snippet = snippet[:MAX_SNIPPET_CHARS] + "…"
        sources.append(snippet)
        combined_for_beautify.append(f"[S{i}] {snippet}")

    beautify_input = "\n\n".join(combined_for_beautify)
    beautified_aggregate = beautify_text(beautify_input)

    # Validate on the aggregated context (NOT just first doc)
    validate_response = validator(beautified_aggregate, query).lower()
    if "yes" in validate_response:
        print("i can generate answer")

        # Get last 5 chat history for this conversation
        from conversation_state import conversation_states
        state = conversation_states.get(id, {})
        chat_history = state.get("chat_history", [])

        # Build strict, source-aware prompt
        prompt = build_prompt(query, grouped, beautified_aggregate, chat_history, sources)

        # Call LLM
        try:
            print("Prompt for Mistral:")
            response = ollama.chat(
                model="mistral",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Respond naturally."},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.2}
            )
            response_text = response["message"]["content"].strip()
            print("Mistral response:", response_text)

            # Update chat history (keep last 5)
            from conversation_state import update_chat_history
            update_chat_history(id, query, response_text)
            return {"response": response_text}

        except Exception as e:
            raise HTTPException(500, str(e))

    # If validator says no, abstain clearly
    abstain = (
        "The context is out of my knowledge. "
        "Try rephrasing, or provide more specific details/keywords."
    )
    return {"response": abstain}

#============================OLD UTILS BELOW============================
# import re
# import ollama
# from chromadb import PersistentClient
# from embedding_model import get_embedder
# from pathlib import Path
# from fastapi import HTTPException

# WORK_ROOT = Path("/home/ddata/dev/nexusFlow-backend/service/rag_workspace")
# CHROMA_DB_DIR = str(WORK_ROOT / "dev_chromadb")
# COLLECTION_NAME = "dev_embeddings"
# SIMILARITY_THRESHOLD = 0.35

# def is_greeting(text):
#     """Check if the user input is a greeting."""
#     return bool(re.search(r"\b(hi+|hello+|hey+|good (morning|afternoon|evening))\b", text.lower()))

# def build_prompt(query: str, context: dict, beautified_text, chat_history=None) -> str:
#     """
#     Builds a structured and constrained prompt for the LLM to avoid hallucination, including last 5 chat history.
#     """
#     prompt = (
#         "You are a highly accurate assistant. "
#         "Only answer questions using the information provided in the context below. "
#         "**If you cannot find the answer in the context, say: 'The context is out of my knowledge.'** "
#         "Do not guess or assume anything outside of the context.\n\n"
#     )
#     # Add chat history if available
#     if chat_history:
#         prompt += "=== CHAT HISTORY (last 5) ===\n"
#         for turn in chat_history:
#             prompt += f"User: {turn['question']}\nAssistant: {turn['answer']}\n"
#         prompt += "\n"
#     # Add beautified text
#     if beautified_text:
#         prompt += f"=== BEAUTIFIED TEXT ===\n{beautified_text}\n\n"
#     # Add table summaries
#     if context.get("table"):
#         prompt += "=== TABLES ===\n"
#         for table, meta in context["table"]:
#             page = meta.get("page", "unknown")
#             prompt += f"- Page {page}: {table.strip()}\n"
#         prompt += "\n"
#     # Add image captions
#     if context.get("image"):
#         prompt += "=== IMAGES ===\n"
#         for image_caption, meta in context["image"]:
#             prompt += f"- {image_caption} (image_id: {meta.get('id')})\n"
#         prompt += "\n"
#     # Final instruction with beautified query
#     prompt += (
#         f"=== QUESTION ===\n{query}\n\n"
#         "=== ANSWER ==="
#     )
#     return prompt

# def validator(beautify_data, query):
#     prompt = (
#         "You are an validation specialist. "
#         "where you understand the given data and validate it and you say it can able to answer the query given by the user."
#         f"\n\nContext to verify :\n{beautify_data}\n\n"
#         f"\n\n query :\n{query}\n\n"
#         "just say 'yes' if it can answer the query, otherwise say 'no'. "
#         ""
#     )
#     response = ollama.chat(
#             model="mistral",
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant. Respond naturally."},
#                 {"role": "user", "content": prompt}
#             ],
#             options={"temperature": 0.4}
#         )
#     response_text = response["message"]["content"].strip()
#     print("--------------------------------")
#     print("validator response:", response_text)
#     print("--------------------------------")
#     return response_text

# def beautify_text(data):
#     prompt = (
#         "you are a inspection specialist. "
#         " where you inspect the given data and format it in a nice way to pass the data to next step."
#         " which helps in order to generate a answer from it."
#         " make sure dont change any context , just format and beautify."
#         f"\n\nInput text:\n{data}\n\n"
#     )
#     response = ollama.chat(
#             model="mistral",
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant. Respond naturally."},
#                 {"role": "user", "content": prompt}
#             ],
#             options={"temperature": 0.4}
#         )
#     response_text = response["message"]["content"].strip()
#     print("--------------------------------")
#     print("beautify response:", response_text)
#     print("--------------------------------")
#     return response_text

# def chat_with_knowledge(query, id):
#     print("Query received", query)
#     if is_greeting(query):
#         return {"response": "Hi there! How can I assist you today?"}

#     # chANGE THE INIT TO TOP
#     client = PersistentClient(path=CHROMA_DB_DIR)
#     coll = client.get_collection(COLLECTION_NAME)
#     # Embed query
#     query_vec = get_embedder().encode([query])[0]
#     # Retrieve from ChromaDB
#     res = coll.query(
#         query_embeddings=[query_vec],
#         n_results=5,
#         include=["documents", "metadatas", "distances"]
#     )
#     print("Ask endpoint response:", res['documents'])

#     if not res["documents"] or not res["documents"][0]:
#         raise HTTPException(404, "No context found")
#     # Filter by similarity
#     beautify_data = beautify_text(res["documents"][0])
#     print("Beautified data:", beautify_data)
#     validate_response = validator(beautify_data, query)
#     if "yes" in validate_response.lower():
#         print("i can generate answer")
#         grouped = {"text": [], "table": [], "image": []}

#         filtered = []
#         for i, doc in enumerate(res["documents"][0]):
#             dist = res["distances"][0][i]
#             print("Distance for doc", i, ":", dist)
#             filtered.append({
#                 "doc": doc,
#                 "meta": res["metadatas"][0][i],
#             })

#         for item in filtered:
#             doc = item["doc"]
#             meta = item["meta"]
#             t = meta.get("type", "text")
#             grouped[t].append((doc, meta))
#         # Get last 5 chat history for this conversation
#         # changr the import statements
#         # check the state of conversation
#         from conversation_state import conversation_states
#         state = conversation_states.get(id, {})
#         chat_history = state.get("chat_history", [])
#         # Build prompt with chat history
#         prompt = build_prompt(query, grouped, beautify_data, chat_history)
#         # Call LLM
#         try:
#             print("Prompt for Mistral:")
#             response = ollama.chat(
#                 model="mistral",
#                 messages=[
#                     {"role": "system", "content": "You are a helpful assistant. Respond naturally."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 options={"temperature": 0.4}
#             )
#             response_text = response["message"]["content"].strip()
#             print("Mistral response:", response_text)
#             # Update chat history
#             # move to up
#             from conversation_state import update_chat_history
#             update_chat_history(id, query, response_text)
#             return {"response": response_text}
#         except Exception as e:
#             raise HTTPException(500, str(e))
#     elif "no" in validate_response.lower():
#         prompt = (
#             "It's seems like it's out of my knowledge or i can't get the details clearly "
#             " I can't answer! "
#             "Please ask something about the knowledge which is updated."
#         )
#         return {"response": prompt}