import numpy as np

from models.schemas import DocumentChunk, SearchResult
from retrieval.bm25_search import BM25Search
from retrieval.vector_store import FAISSVectorStore
from utils.config import settings


def _reciprocal_rank_fusion(
    *ranked_lists: list[tuple[DocumentChunk, float]],
    rrf_k: int = 60,
) -> list[tuple[DocumentChunk, float]]:
    """
    RRF: score(d) = Σ  1 / (rrf_k + rank_i(d))

    Higher is better. Naturally combines lists of different scales.
    """
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, DocumentChunk] = {}

    for ranked in ranked_lists:
        for rank, (chunk, _) in enumerate(ranked, start=1):
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + 1.0 / (rrf_k + rank)
            chunk_map[chunk.id] = chunk

    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
    return [(chunk_map[cid], rrf_scores[cid]) for cid in sorted_ids]


class HybridSearcher:
    def __init__(self, vector_store: FAISSVectorStore, bm25: BM25Search) -> None:
        self._vs = vector_store
        self._bm25 = bm25

    async def search(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> list[SearchResult]:
        fetch_k = top_k * 4  # over-fetch so RRF has more to fuse

        vector_hits = self._vs.search(query_embedding, fetch_k)
        bm25_hits = self._bm25.search(query, fetch_k)

        fused = _reciprocal_rank_fusion(vector_hits, bm25_hits)

        return [
            SearchResult(chunk=chunk, score=score, rank=rank)
            for rank, (chunk, score) in enumerate(fused[:top_k], start=1)
        ]
