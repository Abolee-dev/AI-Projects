from fastapi import APIRouter, Depends

from container import get_ingestion_pipeline
from ingestion.pipeline import IngestionPipeline
from models.schemas import IngestRequest, IngestResponse

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


@router.post("/", response_model=IngestResponse)
async def ingest(
    request: IngestRequest,
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> IngestResponse:
    return await pipeline.ingest(request)


@router.post("/default", response_model=IngestResponse)
async def ingest_default(
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> IngestResponse:
    """Ingest all PDFs from data/documents/."""
    return await pipeline.ingest_default_dir()
