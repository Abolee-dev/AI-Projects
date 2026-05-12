import pytest
from app.security.sql_guardrails import SQLGuardrails, SQLGuardrailError

def test_select_allowed():
    guard = SQLGuardrails()
    sql = guard.validate("SELECT * FROM vulnerability_events")
    assert "LIMIT" in sql

def test_delete_blocked():
    guard = SQLGuardrails()
    with pytest.raises(SQLGuardrailError):
        guard.validate("DELETE FROM vulnerability_events")
