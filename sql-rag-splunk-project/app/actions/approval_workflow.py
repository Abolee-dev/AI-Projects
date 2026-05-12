def requires_human_approval(action_type: str) -> bool:
    return action_type in {"send_email", "create_jira", "servicenow_update"}
