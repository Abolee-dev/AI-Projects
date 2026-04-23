from fastapi import APIRouter

from container import get_bm25, get_llm, get_vector_store
from insights.engine import InsightEngine
from models.schemas import InsightRequest, InsightResponse

router = APIRouter(prefix="/insights", tags=["Insights"])


@router.post("/", response_model=InsightResponse)
async def get_insights(request: InsightRequest) -> InsightResponse:
    engine = InsightEngine(get_vector_store(), get_bm25(), get_llm())
    return await engine.generate(request)
