from app.utils.logger import get_logger

logger = get_logger(__name__)

def audit_event(user_id: str | None, action: str, detail: str):
    logger.info("AUDIT user=%s action=%s detail=%s", user_id or "anonymous", action, detail)
