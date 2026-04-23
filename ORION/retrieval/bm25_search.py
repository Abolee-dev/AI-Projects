import pickle
import re

from rank_bm25 import BM25Okapi

from models.schemas import DocumentChunk
from utils.config import settings
from utils.logger import logger

# Regulatory stopwords — kept minimal intentionally
_STOPWORDS = frozenset(
    {"the", "a", "an", "and", "or", "of", "to", "in", "for", "is", "are", "that"}
)


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


class BM25Search:
    def __init__(self, index_name: str = "main") -> None:
        self._path = settings.bm25_dir / f"{index_name}_bm25.pkl"
        self._bm25: BM25Okapi | None = None
        self._chunks: list[DocumentChunk] = []

    def load(self) -> bool:
        if self._path.exists():
            with open(self._path, "rb") as f:
                data = pickle.load(f)
            self._bm25 = data["bm25"]
            self._chunks = data["chunks"]
            logger.info(f"BM25: loaded {len(self._chunks)} documents")
            return True
        return False

    def build(self, chunks: list[DocumentChunk]) -> None:
        self._chunks = chunks
        corpus = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(corpus)
        self._save()
        logger.info(f"BM25: indexed {len(chunks)} documents")

    def search(self, query: str, k: int) -> list[tuple[DocumentChunk, float]]:
        if not self._bm25 or not self._chunks:
            return []
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            (self._chunks[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]

    def _save(self) -> None:
        settings.bm25_dir.mkdir(parents=True, exist_ok=True)
        with open(self._path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "chunks": self._chunks}, f)
