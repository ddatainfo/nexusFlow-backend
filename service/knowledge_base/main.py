import os
import uuid
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
from dotenv import load_dotenv
import logging
from conversation_state import get_conversation_id, load_conversation_state
from utils import chat_with_knowledge

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
WORK_ROOT = Path("/home/ddata/work/nantha/chatbot_api/pdf_service/rag_workspace")
CONVERSATION_STATE_FILE = "/home/ddata/Interns_work/AI_chatbot/chatbot/dev_conversation_state.pkl"

UPLOAD_DIR = WORK_ROOT / "uploads"
ASSET_DIR = WORK_ROOT / "assets"
CHROMA_DB_DIR = str(WORK_ROOT / "dev_chromadb")
COLLECTION_NAME = "dev_embeddings"

# Initialize FastAPI app
app = FastAPI(title="AI Chatbot with Jira Integration")

# Add SessionMiddleware for conversation ID
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET_KEY", str(uuid.uuid4())),
    session_cookie="session_id",
    max_age=3600  # 1 hour session expiry
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/chat")
async def chat(request: Request, user_input: str):
    conversation_id = get_conversation_id(request)
    response = chat_with_knowledge(user_input, conversation_id)
    return JSONResponse(response)

# Exception handler
@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error at {request.url}: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# Initialize conversation states
conversation_states = load_conversation_state()

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)