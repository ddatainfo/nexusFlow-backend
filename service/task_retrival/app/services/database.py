from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.config.settings import DATABASE_URL
import logging

logger = logging.getLogger(__name__)

# Database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def get_db():
    """Provide a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()