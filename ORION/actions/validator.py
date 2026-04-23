from typing import Any

from models.schemas import ActionPayload
from actions.base import BaseAction
from actions.jira_action import JiraAction
from actions.email_action import EmailAction
from utils.logger import logger

_REGISTRY: list[BaseAction] = [JiraAction(), EmailAction()]


def get_handler(action_type: str) -> BaseAction | None:
    for handler in _REGISTRY:
        if handler.can_handle(action_type):
            return handler
    return None


async def validate_and_execute(
    payload: ActionPayload, auto_execute: bool = False
) -> dict[str, Any]:
    """
    Validates that a handler exists for the action type.
    Only executes if auto_execute=True AND confidence >= 0.8.
    Otherwise returns the payload for human review.
    """
    handler = get_handler(payload.action_type.value)
    if handler is None:
        return {"status": "no_handler", "action_type": payload.action_type}

    if payload.requires_approval and not auto_execute:
        return {"status": "pending_approval", "payload": payload.model_dump()}

    if payload.confidence < 0.8:
        logger.warning(f"Low confidence {payload.confidence:.2f} — skipping auto-execution")
        return {"status": "low_confidence", "confidence": payload.confidence}

    result = await handler.execute(payload)
    return {"status": "executed", "result": result}
