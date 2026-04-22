from fastapi import APIRouter
from app.models.schemas import ChatRequest, ChatResponse
from app.retrieval.hybrid_search import HybridRetriever
from app.llm.answer_generator import AnswerGenerator
from app.llm.prompts import ANSWER_PROMPT

router = APIRouter()


class DummyVectorStore:
    def search(self, query, filters=None, top_k=10):
        return []


class DummyKeywordStore:
    def search(self, query, filters=None, top_k=10):
        return []


class DummyReranker:
    def rerank(self, query, results):
        return results


class DummyLLMClient:
    def generate(self, prompt: str) -> str:
        return "This is a placeholder answer generated from retrieved regulatory context."


retriever = HybridRetriever(DummyVectorStore(), DummyKeywordStore(), DummyReranker())
answer_generator = AnswerGenerator(DummyLLMClient(), ANSWER_PROMPT)


@router.post("/", response_model=ChatResponse)
def chat(request: ChatRequest):
    filters = {
        "regulator": request.regulator,
        "jurisdiction": request.jurisdiction
    }

    chunks = retriever.search(request.question, filters=filters, top_k=request.top_k)
    result = answer_generator.answer(request.question, chunks)

    return ChatResponse(
        answer=result["answer"],
        citations=result["citations"]
    )
