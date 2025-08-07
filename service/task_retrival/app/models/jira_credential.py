from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class JiraCredential(Base):
    __tablename__ = "Credentials"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String)
    app_name = Column(String)
    app_domain = Column(String)
    app_project_key = Column(String)
    app_username = Column(String)
    app_token = Column(String)
    user_id = Column(Integer)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)