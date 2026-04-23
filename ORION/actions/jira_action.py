from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from actions.base import BaseAction
from models.schemas import ActionPayload
from utils.config import settings
from utils.logger import logger


class JiraAction(BaseAction):
    def can_handle(self, action_type: str) -> bool:
        return action_type == "create_jira"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def execute(self, payload: ActionPayload) -> dict[str, Any]:
        if not all([settings.jira_server, settings.jira_email, settings.jira_api_token]):
            raise ValueError("Jira credentials not configured — set JIRA_* env vars.")

        import asyncio
        from jira import JIRA

        p = payload.payload
        loop = asyncio.get_event_loop()

        def _create():
            jira = JIRA(
                server=settings.jira_server,
                basic_auth=(settings.jira_email, settings.jira_api_token),
            )
            issue = jira.create_issue(
                project=settings.jira_project_key,
                summary=p.get("summary", "Regulatory finding"),
                description=p.get("description", ""),
                issuetype={"name": "Task"},
                priority={"name": p.get("priority", "Medium")},
            )
            return issue.key

        issue_key = await loop.run_in_executor(None, _create)
        logger.info(f"Jira issue created: {issue_key}")
        return {"issue_key": issue_key, "server": settings.jira_server}
