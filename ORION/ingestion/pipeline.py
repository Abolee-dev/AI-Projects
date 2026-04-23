import time
from pathlib import Path

from ingestion.chunker import chunk_pages
from ingestion.pdf_loader import load_directory, load_pdf
from models.schemas import DocumentChunk, IngestRequest, IngestResponse
from utils.config import settings
from utils.logger import logger


class IngestionPipeline:
    """Orchestrates PDF → chunks, then hands off to retrieval for indexing."""

    def __init__(self, vector_store, bm25_search, embedder):
        self._vector_store = vector_store
        self._bm25 = bm25_search
        self._embedder = embedder

    async def ingest(self, request: IngestRequest) -> IngestResponse:
        start = time.time()
        errors: list[str] = []
        all_chunks: list[DocumentChunk] = []

        paths: list[Path] = []
        if request.source_dir:
            paths_dir = Path(request.source_dir)
            if not paths_dir.exists():
                errors.append(f"Directory not found: {request.source_dir}")
            else:
                pages = load_directory(paths_dir)
                all_chunks.extend(chunk_pages(pages))

        for fp in request.file_paths:
            p = Path(fp)
            if not p.exists():
                errors.append(f"File not found: {fp}")
                continue
            try:
                pages = load_pdf(p)
                all_chunks.extend(chunk_pages(pages))
            except Exception as e:
                errors.append(f"Failed to process {fp}: {e}")

        processed_files = len(request.file_paths) + (1 if request.source_dir else 0)

        if all_chunks:
            logger.info(f"Embedding {len(all_chunks)} chunks…")
            texts = [c.text for c in all_chunks]
            embeddings = await self._embedder.embed_texts(texts)
            self._vector_store.add(all_chunks, embeddings)
            self._bm25.build(all_chunks)
            logger.info("Indexing complete.")

        return IngestResponse(
            processed=processed_files - len(errors),
            chunks_created=len(all_chunks),
            errors=errors,
            duration_seconds=round(time.time() - start, 2),
        )

    async def ingest_default_dir(self) -> IngestResponse:
        settings.ensure_dirs()
        return await self.ingest(IngestRequest(source_dir=str(settings.documents_dir)))
