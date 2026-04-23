import pickle
from pathlib import Path

import faiss
import numpy as np

from models.schemas import DocumentChunk
from utils.config import settings
from utils.logger import logger


class FAISSVectorStore:
    def __init__(self, index_name: str = "main") -> None:
        self._index_path = settings.index_dir / f"{index_name}.faiss"
        self._meta_path = settings.index_dir / f"{index_name}.pkl"
        self._index: faiss.Index | None = None
        self._chunks: list[DocumentChunk] = []

    # ── Persistence ───────────────────────────────────────────────────────────

    def load(self) -> bool:
        if self._index_path.exists() and self._meta_path.exists():
            self._index = faiss.read_index(str(self._index_path))
            with open(self._meta_path, "rb") as f:
                self._chunks = pickle.load(f)
            logger.info(f"FAISS: loaded {len(self._chunks)} chunks from disk")
            return True
        return False

    def _save(self) -> None:
        settings.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._index_path))
        with open(self._meta_path, "wb") as f:
            pickle.dump(self._chunks, f)

    # ── Build / Add ───────────────────────────────────────────────────────────

    def build(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> None:
        dim = embeddings.shape[1]
        n = len(chunks)
        if n > 10_000:
            nlist = min(n // 100, 256)
            quantizer = faiss.IndexFlatIP(dim)
            self._index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            faiss.normalize_L2(embeddings)
            self._index.train(embeddings)
        else:
            self._index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(embeddings)

        self._index.add(embeddings)
        self._chunks = chunks
        self._save()
        logger.info(f"FAISS: built index with {n} vectors (dim={dim})")

    def add(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> None:
        if self._index is None:
            self.build(chunks, embeddings)
            return
        faiss.normalize_L2(embeddings)
        self._index.add(embeddings)
        self._chunks.extend(chunks)
        self._save()
        logger.info(f"FAISS: added {len(chunks)} vectors (total={self.size})")

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self, query_embedding: np.ndarray, k: int
    ) -> list[tuple[DocumentChunk, float]]:
        if not self._index or self._index.ntotal == 0:
            return []
        q = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q)
        scores, indices = self._index.search(q, min(k, self._index.ntotal))
        return [
            (self._chunks[idx], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0
        ]

    @property
    def size(self) -> int:
        return len(self._chunks)
