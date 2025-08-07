import uvicorn
from fastapi import FastAPI
from app.routes.chat import router as chat_router
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s │ %(levelname)s │ %(name)s │ %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Jira Chat & Task Retrieval Service")

# Include routes
app.include_router(chat_router, prefix="/chat")

if __name__ == "__main__":
    logger.info("Starting Jira Chat & Task Retrieval Service on port 8003")
    uvicorn.run(app, host="0.0.0.0", port=8003)