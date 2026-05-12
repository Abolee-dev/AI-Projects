from app.agents.intent_router import IntentRouter
from app.agents.sql_generator_agent import SQLGeneratorAgent
from app.agents.summarizer_agent import SummarizerAgent
from app.agents.email_draft_agent import EmailDraftAgent
from app.database.schema_registry import SchemaRegistry
from app.database.query_executor import QueryExecutor
from app.security.sql_guardrails import SQLGuardrails
from app.security.pii_masking import mask_rows
from app.security.audit_logger import audit_event

class QueryService:
    def __init__(self):
        self.intent_router = IntentRouter()
        self.schema_registry = SchemaRegistry()
        self.sql_generator = SQLGeneratorAgent()
        self.sql_guardrails = SQLGuardrails()
        self.query_executor = QueryExecutor()
        self.summarizer = SummarizerAgent()
        self.email_agent = EmailDraftAgent()

    def handle_query(self, question: str, user_id: str | None = None) -> dict:
        intent = self.intent_router.classify(question)
        schema_text = self.schema_registry.all_schema_text()

        sql = self.sql_generator.generate(question, schema_text)
        safe_sql = self.sql_guardrails.validate(sql)

        audit_event(user_id, "execute_sql", safe_sql)

        rows = self.query_executor.execute(safe_sql)
        rows = mask_rows(rows)

        answer = self.summarizer.summarize(question, rows)

        response = {
            "intent": intent,
            "sql": safe_sql,
            "answer": answer,
            "rows": rows,
            "email_draft": None,
            "requires_approval": False
        }

        if intent == "draft_email":
            response["email_draft"] = self.email_agent.draft(question, rows)
            response["requires_approval"] = True

        if intent == "create_ticket":
            response["answer"] += "\n\nTicket creation is intentionally disabled in this prototype. Add Jira approval workflow before enabling."
            response["requires_approval"] = True

        return response
