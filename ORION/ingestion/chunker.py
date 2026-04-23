import hashlib
import re
from dataclasses import dataclass

from ingestion.pdf_loader import RawPage
from models.schemas import DocumentChunk
from utils.config import settings


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter — keeps regulatory numbering intact."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_page(page: RawPage, chunk_size: int, overlap: int) -> list[DocumentChunk]:
    """Split a single page into overlapping chunks by word count."""
    words = page.text.split()
    chunks: list[DocumentChunk] = []
    step = max(1, chunk_size - overlap)
    idx = 0

    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if not window:
            break
        text = " ".join(window)
        chunk_id = hashlib.sha256(
            f"{page.source}:{page.page}:{idx}".encode()
        ).hexdigest()[:16]
        doc_id = hashlib.sha256(page.source.encode()).hexdigest()[:12]

        chunks.append(
            DocumentChunk(
                id=chunk_id,
                doc_id=doc_id,
                source=page.source,
                page=page.page,
                chunk_index=idx,
                text=text,
            )
        )
        idx += 1

    return chunks


def chunk_pages(pages: list[RawPage]) -> list[DocumentChunk]:
    all_chunks: list[DocumentChunk] = []
    for page in pages:
        all_chunks.extend(
            chunk_page(page, settings.chunk_size, settings.chunk_overlap)
        )
    return all_chunks
