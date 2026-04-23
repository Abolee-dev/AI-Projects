import asyncio
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from actions.base import BaseAction
from models.schemas import ActionPayload
from utils.config import settings
from utils.logger import logger


class EmailAction(BaseAction):
    def can_handle(self, action_type: str) -> bool:
        return action_type == "send_email"

    async def execute(self, payload: ActionPayload) -> dict[str, Any]:
        if not settings.smtp_host:
            raise ValueError("SMTP not configured — set SMTP_* env vars.")

        p = payload.payload
        recipient = p.get("to", "")
        subject = p.get("subject", "Regulatory Alert — ORION")
        body = p.get("body", "")

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._send, recipient, subject, body)
        logger.info(f"Email sent to {recipient}")
        return {"status": "sent", "recipient": recipient}

    def _send(self, to: str, subject: str, body: str) -> None:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = settings.smtp_from
        msg["To"] = to
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
            server.starttls()
            server.login(settings.smtp_user, settings.smtp_password)
            server.sendmail(settings.smtp_from, [to], msg.as_string())
