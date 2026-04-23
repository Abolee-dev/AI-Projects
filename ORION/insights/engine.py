import re
from collections import Counter

from llm.client import LLMClient
from models.schemas import InsightRequest, InsightResponse, TopicFrequency
from retrieval.bm25_search import BM25Search
from retrieval.vector_store import FAISSVectorStore

# Regulatory topic seeds — extend as needed
_TOPIC_SEEDS = [
    "AML", "KYC", "outsourcing", "SYSC", "operational risk", "credit risk",
    "liquidity", "capital", "conduct", "GDPR", "data protection", "sanctions",
    "market abuse", "MiFID", "PSD2", "Basel", "ICAAP", "stress testing",
    "third-party risk", "vendor risk", "cyber", "fraud",
]


class InsightEngine:
    def __init__(
        self, vector_store: FAISSVectorStore, bm25: BM25Search, llm: LLMClient
    ) -> None:
        self._vs = vector_store
        self._bm25 = bm25
        self._llm = llm

    async def generate(self, request: InsightRequest) -> InsightResponse:
        chunks = self._vs._chunks  # read corpus
        if not chunks:
            return InsightResponse(summary="No documents indexed yet.", topics=[])

        # Deterministic analytics first
        topic_counts = self._count_topics(chunks)
        top_topics = topic_counts.most_common(request.limit)

        topics: list[TopicFrequency] = []
        for topic, count in top_topics:
            sources = list(
                {c.source for c in chunks if topic.lower() in c.text.lower()}
            )[:5]
            topics.append(TopicFrequency(topic=topic, count=count, sources=sources))

        # LLM narrative summary — uses analytics output, not raw chunks
        topic_lines = "\n".join(f"- {t.topic}: {t.count} mentions" for t in topics)
        prompt = (
            f"Summarise regulatory focus areas based on these topic frequencies:\n"
            f"{topic_lines}\n\nWrite 3-4 sentences for a compliance officer."
        )
        summary = await self._llm.complete(
            system="You are a regulatory analytics assistant.", user=prompt
        )

        return InsightResponse(summary=summary, topics=topics)

    def _count_topics(self, chunks) -> Counter:
        counter: Counter = Counter()
        for chunk in chunks:
            text_lower = chunk.text.lower()
            for seed in _TOPIC_SEEDS:
                if seed.lower() in text_lower:
                    counter[seed] += 1
        return counter
