from abc import ABC, abstractmethod

import numpy as np

from utils.config import EmbeddingProvider, settings
from utils.logger import logger


class BaseEmbedder(ABC):
    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    async def embed_query(self, query: str) -> np.ndarray: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...


# ── OpenAI ────────────────────────────────────────────────────────────────────

class OpenAIEmbedder(BaseEmbedder):
    def __init__(self) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_embedding_model
        # text-embedding-3-small → 1536, text-embedding-3-large → 3072
        self._dim = 3072 if "large" in self._model else 1536

    async def embed_texts(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = await self._client.embeddings.create(model=self._model, input=batch)
            all_embeddings.extend(d.embedding for d in resp.data)
        return np.array(all_embeddings, dtype=np.float32)

    async def embed_query(self, query: str) -> np.ndarray:
        arr = await self.embed_texts([query])
        return arr[0]

    @property
    def dimension(self) -> int:
        return self._dim


# ── Qwen ─────────────────────────────────────────────────────────────────────

class QwenEmbedder(BaseEmbedder):
    """
    Two modes:
      - Local  : loads Qwen model via sentence-transformers (default)
      - API    : uses DashScope-compatible OpenAI endpoint when QWEN_API_KEY is set
    """

    def __init__(self) -> None:
        self._use_api = bool(settings.qwen_api_key and settings.qwen_api_base)
        if self._use_api:
            self._init_api()
        else:
            self._init_local()

    def _init_local(self) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = SentenceTransformer(
            settings.qwen_model_name,
            device=device,
            trust_remote_code=True,
        )
        self._dim = self._model.get_sentence_embedding_dimension()
        logger.info(f"Qwen local embedder on {device}, dim={self._dim}")

    def _init_api(self) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_api_base,
        )
        self._dim = settings.embedding_dimension
        logger.info("Qwen API embedder initialised")

    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        if self._use_api:
            return await self._embed_api(texts)
        return await self._embed_local(texts)

    async def _embed_local(self, texts: list[str]) -> np.ndarray:
        import asyncio

        loop = asyncio.get_event_loop()
        arr = await loop.run_in_executor(
            None,
            lambda: self._model.encode(texts, normalize_embeddings=True),
        )
        return np.array(arr, dtype=np.float32)

    async def _embed_api(self, texts: list[str]) -> np.ndarray:
        resp = await self._client.embeddings.create(
            model=settings.qwen_model_name, input=texts
        )
        return np.array([d.embedding for d in resp.data], dtype=np.float32)

    async def embed_query(self, query: str) -> np.ndarray:
        arr = await self.embed_texts([query])
        return arr[0]

    @property
    def dimension(self) -> int:
        return self._dim


# ── Factory ───────────────────────────────────────────────────────────────────

def get_embedder() -> BaseEmbedder:
    if settings.embedding_provider == EmbeddingProvider.QWEN:
        return QwenEmbedder()
    return OpenAIEmbedder()
