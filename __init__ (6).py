from typing import List, Dict


class AnswerGenerator:
    def __init__(self, llm_client, prompt_template: str):
        self.llm_client = llm_client
        self.prompt_template = prompt_template

    def build_context(self, chunks: List[Dict]) -> str:
        parts = []
        for i, c in enumerate(chunks, start=1):
            parts.append(
                f"[{i}] Title: {c.get('title', '')}\n"
                f"Section: {c.get('section_heading', '')}\n"
                f"Page: {c.get('page_number', '')}\n"
                f"Text: {c.get('text', '')}"
            )
        return "\n\n".join(parts)

    def answer(self, question: str, chunks: List[Dict]) -> Dict:
        context = self.build_context(chunks)
        prompt = self.prompt_template.format(question=question, context=context)

        response = self.llm_client.generate(prompt)

        citations = [
            {
                "document_id": c["document_id"],
                "title": c.get("title"),
                "page_number": c.get("page_number"),
                "section_heading": c.get("section_heading"),
                "chunk_id": c["chunk_id"]
            }
            for c in chunks
        ]

        return {
            "answer": response,
            "citations": citations
        }
