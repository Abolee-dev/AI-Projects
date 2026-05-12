from app.agents.intent_router import IntentRouter

def test_email_intent():
    router = IntentRouter()
    assert router.classify("draft email for overdue vulnerabilities") == "draft_email"

def test_data_query_intent():
    router = IntentRouter()
    assert router.classify("show critical vulnerabilities") == "data_query"
