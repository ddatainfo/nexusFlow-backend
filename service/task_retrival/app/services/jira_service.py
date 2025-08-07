from typing import Optional, List
from fastapi import HTTPException
from sqlalchemy.orm import Session
from app.models.jira_credential import JiraCredential
import requests
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def get_jira_details(user_id: int, db: Session) -> dict:
    logger.debug(f"Fetching Jira credentials for user_id={user_id}")
    record = db.query(JiraCredential).filter(JiraCredential.user_id == user_id).first()
    if not record:
        logger.error(f"No Jira credentials found for user_id={user_id}")
        raise HTTPException(status_code=404, detail=f"No Jira credentials for user_id={user_id}")
    logger.debug(f"Credentials found: username={record.app_username}, domain={record.app_domain}")
    return {
        "jira_email": record.app_username,
        "jira_domain": record.app_domain,
        "jira_project_key": record.app_project_key,
        "jira_api_token": record.app_token,
    }

def get_account_id_from_name(creds: dict, user_name: str) -> Optional[str]:
    logger.debug(f"Searching Jira accountId for user_name='{user_name}' on domain '{creds['jira_domain']}'")
    url = f"https://{creds['jira_domain']}/rest/api/3/user/search"
    auth = (creds['jira_email'], creds['jira_api_token'])
    params = {"query": user_name, "maxResults": 1}
    resp = requests.get(url, auth=auth, params=params)
    logger.debug(f"User search response status: {resp.status_code}")
    if resp.status_code != 200:
        logger.error(f"Failed to fetch accountId from Jira: {resp.text}")
        raise HTTPException(status_code=resp.status_code, detail="Failed to fetch accountId from Jira")
    users = resp.json()
    if not users:
        logger.warning(f"No users found in Jira search for '{user_name}'")
        return None
    account_id = users[0].get('accountId')
    logger.debug(f"Found accountId: {account_id} for user_name='{user_name}'")
    return account_id

def run_jira_jql(creds: dict, jql: str, fields: str = "summary,status,created") -> List[dict]:
    logger.debug(f"Running JQL query: {jql}")
    url = f"https://{creds['jira_domain']}/rest/api/3/search"
    auth = (creds['jira_email'], creds['jira_api_token'])
    params = {"jql": jql, "fields": fields, "maxResults": 50}
    resp = requests.get(url, auth=auth, params=params)
    logger.debug(f"Jira search response status: {resp.status_code}")
    if resp.status_code != 200:
        logger.error(f"Jira search failed: {resp.text}")
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    issues = resp.json().get("issues", [])
    logger.debug(f"Number of issues returned: {len(issues)}")
    return issues

def query_tasks_by_username(creds: dict, username: str) -> List[dict]:
    logger.debug(f"Querying tasks assigned to username='{username}'")
    account_id = get_account_id_from_name(creds, username)
    if not account_id:
        logger.error(f"User '{username}' not found in Jira")
        raise HTTPException(status_code=404, detail=f"User '{username}' not found in Jira")
    jql = f'project = {creds["jira_project_key"]} AND assignee = "{account_id}" ORDER BY created DESC'
    return run_jira_jql(creds, jql)

def query_tasks_by_date_and_optional_username(creds: dict, from_date: str, to_date: str, username: Optional[str] = None) -> List[dict]:
    logger.debug(f"Querying tasks from_date='{from_date}', to_date='{to_date}', username='{username}'")
    try:
        if from_date:
            datetime.strptime(from_date, "%Y-%m-%d")
        if to_date:
            datetime.strptime(to_date, "%Y-%m-%d")
    except ValueError:
        logger.error("Invalid date format. Use YYYY-MM-DD")
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    jql_filters = [f'project = {creds["jira_project_key"]}']
    if from_date:
        jql_filters.append(f'created >= "{from_date} 00:00"')
    if to_date:
        jql_filters.append(f'created <= "{to_date} 23:59"')

    if username:
        account_id = get_account_id_from_name(creds, username)
        if not account_id:
            logger.error(f"User '{username}' not found in Jira")
            raise HTTPException(status_code=404, detail=f"User '{username}' not found in Jira")
        jql_filters.append(f'assignee = "{account_id}"')

    jql = " AND ".join(jql_filters) + " ORDER BY created DESC"
    return run_jira_jql(creds, jql)

def query_tasks_by_status_and_optional_username(creds: dict, status: str, username: Optional[str] = None) -> List[dict]:
    logger.debug(f"Querying tasks by status='{status}', username='{username}'")
    status_lower = status.lower()
    jql_filters = [f'project = {creds["jira_project_key"]}']
    if status_lower in ["done", "completed"]:
        jql_filters.append('status = "Done"')
    elif status_lower in ["pending", "open", "to do", "in progress"]:
        jql_filters.append('status IN ("To Do", "In Progress")')
    else:
        jql_filters.append(f'status = "{status}"')

    if username:
        account_id = get_account_id_from_name(creds, username)
        if not account_id:
            logger.error(f"User '{username}' not found in Jira")
            raise HTTPException(status_code=404, detail=f"User '{username}' not found in Jira")
        jql_filters.append(f'assignee = "{account_id}"')

    jql = " AND ".join(jql_filters) + " ORDER BY created DESC"
    return run_jira_jql(creds, jql)