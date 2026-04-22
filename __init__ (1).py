from fastapi import APIRouter
from app.models.schemas import InsightRequest, InsightResponse

router = APIRouter()


@router.post("/", response_model=InsightResponse)
def insights(request: InsightRequest):
    return InsightResponse(
        summary="Placeholder insight summary.",
        key_themes=["Operational resilience", "Third-party risk"],
        trend_points=["References increased over time", "Audit focus shifted toward evidence and testing"]
    )
