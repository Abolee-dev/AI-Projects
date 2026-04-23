import json
import time

import numpy as np

from llm.client import LLMClient
from llm.prompts import (
    ACTION_EXTRACTION_PROMPT,
    SYSTEM_PROMPT,
    build_rag_prompt,
)
from models.schemas import (
    ActionPayload,
    ActionType,
    Citation,
    QueryRequest,
    QueryResponse,
    SearchResult,
)
from retrieval.embeddings import BaseEmbedder
from retrieval.hybrid_search import HybridSearcher
from retrieval.reranker import CrossEncoderReranker
from utils.config import settings
from utils.logger import logger


class RAGChain:
    def __init__(
        self,
        embedder: BaseEmbedder,
        searcher: HybridSearcher,
        reranker: CrossEncoderReranker,
        llm: LLMClient,
    ) -> None:
        self._embedder = embedder
        self._searcher = searcher
        self._reranker = reranker
        self._llm = llm

    async def run(self, request: QueryRequest) -> QueryResponse:
        t0 = time.time()

        # 1. Embed query
        query_emb = await self._embedder.embed_query(request.query)

        # 2. Hybrid search
        candidates = await self._searcher.search(
            query=request.query,
            query_embedding=query_emb,
            top_k=settings.top_k,
        )

        # 3. Rerank
        results: list[SearchResult] = await self._reranker.rerank(
            query=request.query,
            results=candidates,
            top_k=request.top_k,
        )

        if not results:
            logger.warning("No relevant documents found for query.")
            return QueryResponse(
                query=request.query,
                answer="I could not find relevant regulatory information for your query.",
                citations=[],
                latency_ms=round((time.time() - t0) * 1000, 1),
            )

        # 4. Build RAG prompt & generate answer
        user_prompt = build_rag_prompt(request.query, results)
        answer = await self._llm.complete(SYSTEM_PROMPT, user_prompt)

        # 5. Build citations
        citations = [
            Citation(
                source=r.chunk.source,
                page=r.chunk.page,
                chunk_index=r.chunk.chunk_index,
                relevance_score=round(r.score, 4),
            )
            for r in results
        ]

        # 6. Optional action suggestion
        suggested_action: ActionPayload | None = None
        if request.enable_actions:
            suggested_action = await self._extract_action(answer)

        return QueryResponse(
            query=request.query,
            answer=answer,
            citations=citations,
            suggested_action=suggested_action,
            latency_ms=round((time.time() - t0) * 1000, 1),
        )

    async def _extract_action(self, answer: str) -> ActionPayload | None:
        try:
            raw = await self._llm.complete_json(
                system="You extract action recommendations from regulatory answers.",
                user=ACTION_EXTRACTION_PROMPT.format(answer=answer),
            )
            data = json.loads(raw)
            action_type = ActionType(data.get("action_type", "none"))
            if action_type == ActionType.NONE:
                return None
            return ActionPayload(
                action_type=action_type,
                payload=data.get("payload", {}),
                confidence=float(data.get("confidence", 0.5)),
            )
        except Exception as e:
            logger.warning(f"Action extraction failed: {e}")
            return None
