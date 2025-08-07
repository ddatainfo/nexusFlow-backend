import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database settings
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", 3306)
DB_NAME = os.getenv("DB_NAME")

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Ollama Mistral API settings
OLLAMA_BASE_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"