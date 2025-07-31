import os
import requests
import logging
import os.path
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session

# --------------------------
# Load environment variables
# --------------------------
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", 3306)
DB_NAME = os.getenv("DB_NAME")

# --------------------------
# Logging setup
# --------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------
# SQLAlchemy setup
# --------------------------
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --------------------------
# SQLAlchemy model
# --------------------------
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

# --------------------------
# Fetch Jira credentials
# --------------------------
def get_jira_details(user_id: int):
    session: Session = SessionLocal()
    try:
        record = session.query(JiraCredential).filter(JiraCredential.user_id == user_id).first()
        if not record:
            raise ValueError(f"❌ No Jira credentials found for user_id={user_id}")
        logger.debug(f"🎯 Found record for user_id={user_id}: {record.__dict__}")
        return {
            "jira_email": record.app_username,
            "jira_domain": record.app_domain,
            "jira_project_key": record.app_project_key,
            "jira_api_token": record.app_token,
        }
    finally:
        session.close()

# --------------------------
# Create Jira Ticket
# --------------------------
def create_jira_ticket(fields, attachment=None, user_id: int = 7):
    try:
        creds = get_jira_details(user_id)
        logger.debug("🔐 Retrieved Jira credentials:")
        logger.debug(f"JIRA_EMAIL: {creds['jira_email']}")
        logger.debug(f"JIRA_API_TOKEN: {creds['jira_api_token'][:5]}... (truncated)")  # Don't log full token
        logger.debug(f"JIRA_DOMAIN: {creds['jira_domain']}")
        logger.debug(f"JIRA_PROJECT_KEY: {creds['jira_project_key']}")

        JIRA_EMAIL = creds["jira_email"]
        JIRA_DOMAIN = creds["jira_domain"]
        JIRA_PROJECT_KEY = creds["jira_project_key"]
        JIRA_API_TOKEN = creds["jira_api_token"]
        JIRA_BASE_URL = f"https://{JIRA_DOMAIN}"

        if not all([JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN, JIRA_PROJECT_KEY]):
            raise ValueError("❌ One or more JIRA database fields are missing.")

        url = f"{JIRA_BASE_URL}/rest/api/3/issue"
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        auth = (JIRA_EMAIL, JIRA_API_TOKEN)

        data = {
            "fields": {
                "project": {"key": JIRA_PROJECT_KEY},
                "summary": fields.get("title"),
                "description": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [
                                {
                                    "type": "text",
                                    "text": fields.get("description") or ""
                                }
                            ]
                        }
                    ]
                },
                "issuetype": {"name": "Task"},
                "priority": {
                    "name": str(fields.get("priority", "Medium")).strip().lower().capitalize()
                }
            }
        }

        logger.info(f"📤 Creating Jira ticket for user_id={user_id}")
        logger.debug(f"🔗 URL: {url}")
        logger.debug(f"📁 Project: {JIRA_PROJECT_KEY} | Summary: {fields.get('title')}")

        response = requests.post(url, headers=headers, auth=auth, json=data)
        logger.debug(f"🧾 Request JSON: {data}")
        logger.debug(f"🔍 Jira API response code: {response.status_code}")
        logger.debug(f"🔍 Jira API response text: {response.text}")

        logger.info(f"🔍 Jira API response: {response.status_code} {response.text}")

        if response.status_code != 201:
            logger.error(f"❌ Failed to create JIRA ticket: {response.status_code} {response.text}")
            return None, None

        ticket_key = response.json()["key"]
        ticket_url = f"{JIRA_BASE_URL}/browse/{ticket_key}"
        logger.info(f"✅ Ticket created: {ticket_key} — <{ticket_url}>")

        # --------------------------
        # Attachment (optional)
        # --------------------------
        if attachment:
            attach_url = f"{JIRA_BASE_URL}/rest/api/3/issue/{ticket_key}/attachments"
            attach_headers = {
                "X-Atlassian-Token": "no-check"
            }

            files = {}
            if isinstance(attachment, dict) and os.path.isfile(attachment["path"]):
                with open(attachment["path"], "rb") as f:
                    files["file"] = (attachment["filename"], f, attachment["content_type"])
                    upload_resp = requests.post(
                        attach_url, headers=attach_headers, auth=auth, files=files
                    )
            else:
                files["file"] = ("pasted_link.txt", attachment.encode("utf-8"), "text/plain")
                upload_resp = requests.post(
                    attach_url, headers=attach_headers, auth=auth, files=files
                )

            if upload_resp.status_code not in [200, 201]:
                logger.error(f"❌ Attachment upload failed: {upload_resp.text}")
            else:
                logger.info("📎 Attachment uploaded successfully.")

        return ticket_key, ticket_url

    except Exception as e:
        logger.exception(f"❌ Exception during ticket creation: {str(e)}")
        
        return None, None



# import requests
# import os
# from dotenv import load_dotenv

# load_dotenv()

# JIRA_DOMAIN = os.getenv("JIRA_DOMAIN")
# JIRA_BASE_URL = f"https://{JIRA_DOMAIN}" if JIRA_DOMAIN else None
# JIRA_EMAIL = os.getenv("JIRA_EMAIL")
# JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
# JIRA_PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY")

# if not all([JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN, JIRA_PROJECT_KEY]):
#     raise ValueError("❌ One or more JIRA environment variables are missing.")

# def create_jira_ticket(fields, attachment=None):
#     try:
#         url = f"{JIRA_BASE_URL}/rest/api/3/issue"
#         headers = {
#             "Accept": "application/json",
#             "Content-Type": "application/json"
#         }
#         auth = (JIRA_EMAIL, JIRA_API_TOKEN)
#         data = {
#             "fields": {
#                 "project": {"key": JIRA_PROJECT_KEY},
#                 "summary": fields.get("title"),
#                 "description": {
#                     "type": "doc",
#                     "version": 1,
#                     "content": [
#                         {
#                             "type": "paragraph",
#                             "content": [
#                                 {
#                                     "type": "text",
#                                     "text": fields.get("description") or ""
#                                 }
#                             ]
#                         }
#                     ]
#                 },
#                 "issuetype": {"name": "Task"},
#                 "priority": {
#                     "name": str(fields.get("priority", "Medium")).strip().lower().capitalize()
#                 }
#             }
#         }
#         response = requests.post(url, headers=headers, auth=auth, json=data)
#         if response.status_code != 201:
#             print("❌ Failed to create JIRA ticket:", response.text)
#             return None, None

#         ticket_key = response.json()["key"]
#         ticket_url = f"{JIRA_BASE_URL}/browse/{ticket_key}"
#         print(f"\n✅ Ticket created: {ticket_key} — <{ticket_url}>")
        


#         if attachment:
#             attach_url = f"{JIRA_BASE_URL}/rest/api/3/issue/{ticket_key}/attachments"
#             attach_headers = {
#                 "X-Atlassian-Token": "no-check"
#             }

#             files = {}
#             if isinstance(attachment, dict) and os.path.isfile(attachment["path"]):
#                 with open(attachment["path"], "rb") as f:
#                     files["file"] = (attachment["filename"], f, attachment["content_type"])
#                     upload_resp = requests.post(
#                         attach_url, headers=attach_headers, auth=auth, files=files
#                     )
#             else:
#                 files["file"] = ("pasted_link.txt", attachment.encode("utf-8"), "text/plain")
#                 upload_resp = requests.post(
#                     attach_url, headers=attach_headers, auth=auth, files=files
#                 )

#             if upload_resp.status_code not in [200, 201]:
#                 print("❌ Attachment upload failed:", upload_resp.text)
#             else:
#                 print("📎 Attachment uploaded successfully.")

#         return ticket_key, ticket_url

#     except Exception as e:
#         print("❌ Exception during ticket creation:", str(e))
#         return None, None




# import requests
# import os
# from dotenv import load_dotenv

# load_dotenv()

# JIRA_DOMAIN = os.getenv("JIRA_DOMAIN")
# JIRA_BASE_URL = f"https://{JIRA_DOMAIN}" if JIRA_DOMAIN else None
# JIRA_EMAIL = os.getenv("JIRA_EMAIL")
# JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
# JIRA_PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY")

# if not all([JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN, JIRA_PROJECT_KEY]):
#     raise ValueError("❌ One or more JIRA environment variables are missing.")

# def create_jira_ticket(fields, attachment=None):
#     try:
#         url = f"{JIRA_BASE_URL}/rest/api/3/issue"
#         headers = {
#             "Accept": "application/json",
#             "Content-Type": "application/json"
#         }
#         auth = (JIRA_EMAIL, JIRA_API_TOKEN)
#         data = {
#             "fields": {
#                 "project": {"key": JIRA_PROJECT_KEY},
#                 "summary": fields.get("title"),
#                 "description": {
#                     "type": "doc",
#                     "version": 1,
#                     "content": [
#                         {
#                             "type": "paragraph",
#                             "content": [
#                                 {
#                                     "type": "text",
#                                     "text": fields.get("description") or ""
#                                 }
#                             ]
#                         }
#                     ]
#                 },
#                 "issuetype": {"name": "Task"},
#                 "priority": {
#                     "name": str(fields.get("priority", "Medium")).strip().lower().capitalize()
#                 }
#             }
#         }
#         response = requests.post(url, headers=headers, auth=auth, json=data)
#         if response.status_code != 201:
#             print("❌ Failed to create JIRA ticket:", response.text)
#             return None, None

#         ticket_key = response.json()["key"]
#         ticket_url = f"{JIRA_BASE_URL}/browse/{ticket_key}"
#         print(f"\n✅ Ticket created: {ticket_key} — <{ticket_url}>")
        


#         if attachment:
#             attach_url = f"{JIRA_BASE_URL}/rest/api/3/issue/{ticket_key}/attachments"
#             attach_headers = {
#                 "X-Atlassian-Token": "no-check"
#             }

#             files = {}
#             if isinstance(attachment, dict) and os.path.isfile(attachment["path"]):
#                 with open(attachment["path"], "rb") as f:
#                     files["file"] = (attachment["filename"], f, attachment["content_type"])
#                     upload_resp = requests.post(
#                         attach_url, headers=attach_headers, auth=auth, files=files
#                     )
#             else:
#                 files["file"] = ("pasted_link.txt", attachment.encode("utf-8"), "text/plain")
#                 upload_resp = requests.post(
#                     attach_url, headers=attach_headers, auth=auth, files=files
#                 )

#             if upload_resp.status_code not in [200, 201]:
#                 print("❌ Attachment upload failed:", upload_resp.text)
#             else:
#                 print("📎 Attachment uploaded successfully.")

#         return ticket_key, ticket_url

#     except Exception as e:
#         print("❌ Exception during ticket creation:", str(e))
#         return None, None