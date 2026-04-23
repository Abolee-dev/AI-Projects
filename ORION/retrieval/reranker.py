import asyncio

from models.schemas import SearchResult
from utils.logger import logger

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Lazy-loads the cross-encoder so import time stays fast."""

    def __init__(self) -> None:
        self._model = None

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(_MODEL_NAME)
            logger.info(f"Reranker loaded: {_MODEL_NAME}")

    async def rerank(
        self, query: str, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        if not results:
            return []

        self._load()
        pairs = [(query, r.chunk.text) for r in results]
        loop = asyncio.get_event_loop()
        scores: list[float] = await loop.run_in_executor(
            None, self._model.predict, pairs
        )

        ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            SearchResult(chunk=r.chunk, score=float(s), rank=i + 1)
            for i, (r, s) in enumerate(ranked)
        ]
