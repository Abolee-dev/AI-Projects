SYSTEM_PROMPT = """\
You are ORION, a regulatory intelligence assistant.

Rules:
1. Answer ONLY from the provided context. Do NOT use prior knowledge.
2. If the context does not contain enough information, say so explicitly.
3. Every factual claim must reference its source using [source: <filename>, page <N>].
4. Be precise and concise — regulatory professionals value accuracy over verbosity.
5. If the query implies a compliance action (e.g., a risk identified), suggest it clearly.
"""

CONTEXT_TEMPLATE = """\
=== REGULATORY CONTEXT ===
{context}
=========================

Question: {query}

Answer (with citations):"""

ACTION_EXTRACTION_PROMPT = """\
Given this regulatory answer, decide if an action should be triggered.

Answer: {answer}

Respond with JSON only:
{{
  "action_type": "create_jira" | "send_email" | "none",
  "confidence": 0.0-1.0,
  "payload": {{
    "summary": "...",
    "description": "...",
    "priority": "High|Medium|Low"
  }}
}}
"""


def build_context(chunks) -> str:
    parts = []
    for i, result in enumerate(chunks, start=1):
        c = result.chunk
        parts.append(
            f"[{i}] Source: {c.source}, Page {c.page}\n{c.text}"
        )
    return "\n\n".join(parts)


def build_rag_prompt(query: str, chunks) -> str:
    context = build_context(chunks)
    return CONTEXT_TEMPLATE.format(context=context, query=query)
