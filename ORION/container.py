"""
Dependency injection container — shared singleton components.
Both the API and CLI import from here to avoid double-loading models.
"""
from functools import lru_cache

from ingestion.pipeline import IngestionPipeline
from llm.client import LLMClient
from llm.rag_chain import RAGChain
from retrieval.bm25_search import BM25Search
from retrieval.embeddings import BaseEmbedder, get_embedder
from retrieval.hybrid_search import HybridSearcher
from retrieval.reranker import CrossEncoderReranker
from retrieval.vector_store import FAISSVectorStore
from utils.logger import logger


@lru_cache(maxsize=1)
def get_vector_store() -> FAISSVectorStore:
    vs = FAISSVectorStore()
    vs.load()
    return vs


@lru_cache(maxsize=1)
def get_bm25() -> BM25Search:
    bm25 = BM25Search()
    bm25.load()
    return bm25


@lru_cache(maxsize=1)
def get_embedder_singleton() -> BaseEmbedder:
    logger.info("Initialising embedder…")
    return get_embedder()


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()


@lru_cache(maxsize=1)
def get_llm() -> LLMClient:
    return LLMClient()


@lru_cache(maxsize=1)
def get_searcher() -> HybridSearcher:
    return HybridSearcher(get_vector_store(), get_bm25())


@lru_cache(maxsize=1)
def get_rag_chain() -> RAGChain:
    return RAGChain(
        embedder=get_embedder_singleton(),
        searcher=get_searcher(),
        reranker=get_reranker(),
        llm=get_llm(),
    )


@lru_cache(maxsize=1)
def get_ingestion_pipeline() -> IngestionPipeline:
    return IngestionPipeline(
        vector_store=get_vector_store(),
        bm25_search=get_bm25(),
        embedder=get_embedder_singleton(),
    )
