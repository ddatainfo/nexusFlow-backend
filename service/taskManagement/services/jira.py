import requests
import os
from dotenv import load_dotenv

load_dotenv()

JIRA_DOMAIN = os.getenv("JIRA_DOMAIN")
JIRA_BASE_URL = f"https://{JIRA_DOMAIN}" if JIRA_DOMAIN else None
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
JIRA_PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY")

if not all([JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN, JIRA_PROJECT_KEY]):
    raise ValueError("❌ One or more JIRA environment variables are missing.")

def create_jira_ticket(fields, attachment=None):
    try:
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
        response = requests.post(url, headers=headers, auth=auth, json=data)
        if response.status_code != 201:
            print("❌ Failed to create JIRA ticket:", response.text)
            return None, None

        ticket_key = response.json()["key"]
        ticket_url = f"{JIRA_BASE_URL}/browse/{ticket_key}"
        print(f"\n✅ Ticket created: {ticket_key}")

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
                print("❌ Attachment upload failed:", upload_resp.text)
            else:
                print("📎 Attachment uploaded successfully.")

        return ticket_key, ticket_url

    except Exception as e:
        print("❌ Exception during ticket creation:", str(e))
        return None, None