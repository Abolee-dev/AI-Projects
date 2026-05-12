class IntentRouter:
    def classify(self, question: str) -> str:
        q = question.lower()

        if any(word in q for word in ["draft email", "write email", "send email", "mail"]):
            return "draft_email"

        if any(word in q for word in ["executive summary", "bluf", "summary", "md-level", "cfo"]):
            return "executive_summary"

        if any(word in q for word in ["create jira", "ticket", "servicenow"]):
            return "create_ticket"

        return "data_query"
