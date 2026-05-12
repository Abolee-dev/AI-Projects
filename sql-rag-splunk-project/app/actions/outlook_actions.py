def create_email_draft(subject: str, body: str, to: list[str] | None = None) -> dict:
    # Placeholder for Microsoft Graph / Outlook API integration.
    return {
        "status": "draft_created_placeholder",
        "subject": subject,
        "to": to or [],
        "body": body
    }
