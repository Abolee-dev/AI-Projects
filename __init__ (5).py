ANSWER_PROMPT = """
You are a regulatory intelligence assistant.

Answer the user's question only from the provided context.
Do not invent facts.
If the answer is not supported by the context, say so clearly.
Provide a concise but complete answer.
Then provide citations.

User Question:
{question}

Context:
{context}
"""

INSIGHT_PROMPT = """
You are a compliance and audit insight analyst.

Using the evidence below, generate:
1. a concise executive summary
2. key themes over time
3. audit-relevant observations
4. emerging risks or repeated obligations

Do not speculate beyond the evidence.

Evidence:
{evidence}
"""
