from fastapi import APIRouter, HTTPException, Depends
from app.schemas.jira_task import JiraTaskListResponse, ChatRequest
from app.services.jira_service import (
    get_jira_details,
    query_tasks_by_username,
    query_tasks_by_date_and_optional_username,
    query_tasks_by_status_and_optional_username,
    get_account_id_from_name,
    run_jira_jql,
)
from app.services.llm_service import parse_jira_query_with_mistral, resolve_special_dates
from app.services.database import get_db
from sqlalchemy.orm import Session
import logging
from app.schemas.jira_task import JiraTaskListResponse, ChatRequest, JiraTaskItem

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("", response_model=JiraTaskListResponse)
def chat_endpoint(chat_req: ChatRequest, db: Session = Depends(get_db)):
    logger.info(f"Received chat request: user_input='{chat_req.user_input}', user_id={chat_req.user_id}")
    user_input = chat_req.user_input
    user_id = chat_req.user_id or 7

    params = parse_jira_query_with_mistral(user_input)
    username = params.get("username", "").strip()
    status = params.get("status", "").strip()
    from_date = params.get("from_date", "").strip()
    to_date = params.get("to_date", "").strip()

    from_date, to_date = resolve_special_dates(from_date, to_date)

    # If no parameters are extracted, raise an error instead of defaulting to all tasks
    if not username and not status and not from_date and not to_date:
        logger.error("No valid parameters extracted from input, cannot proceed")
        raise HTTPException(status_code=400, detail="Unable to parse input. Please specify a username, status, or date.")

    creds = get_jira_details(user_id, db)

    try:
        if from_date and to_date:
            logger.info(f"Querying by dates from {from_date} to {to_date}, username={username if username else None}, status={status if status else None}")
            if status:
                jql_filters = [
                    f'project = {creds["jira_project_key"]}',
                    f'created >= "{from_date} 00:00"',
                    f'created <= "{to_date} 23:59"'
                ]
                if username:
                    account_id = get_account_id_from_name(creds, username)
                    if not account_id:
                        logger.error(f"User '{username}' not found in Jira")
                        raise HTTPException(status_code=404, detail=f"User '{username}' not found in Jira")
                    jql_filters.append(f'assignee = "{account_id}"')

                status_lower = status.lower()
                if status_lower in ["done", "completed"]:
                    jql_filters.append('status = "Done"')
                elif status_lower in ["pending", "open", "to do", "in progress"]:
                    jql_filters.append('status IN ("To Do", "In Progress")')
                else:
                    jql_filters.append(f'status = "{status}"')

                jql = " AND ".join(jql_filters) + " ORDER BY created DESC"
                issues = run_jira_jql(creds, jql)
            else:
                issues = query_tasks_by_date_and_optional_username(creds, from_date, to_date, username if username else None)

        elif username and not from_date and not to_date and not status:
            logger.info(f"Querying tasks by username: {username}")
            issues = query_tasks_by_username(creds, username)

        elif status:
            logger.info(f"Querying tasks by status: {status}, username={username if username else None}")
            issues = query_tasks_by_status_and_optional_username(creds, status, username if username else None)

        else:
            logger.error("Invalid combination of parameters")
            raise HTTPException(status_code=400, detail="Invalid query parameters")

    except HTTPException as e:
        logger.error(f"Query failed with HTTPException: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error during Jira query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

    logger.info(f"Returning {len(issues)} tasks")
    tasks = [
        JiraTaskItem(
            key=issue["key"],
            summary=issue["fields"].get("summary"),
            status=issue["fields"].get("status", {}).get("name"),
            created=issue["fields"].get("created"),
        )
        for issue in issues
    ]

    return {"tasks": tasks}