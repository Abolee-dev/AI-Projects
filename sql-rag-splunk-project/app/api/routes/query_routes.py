from fastapi import APIRouter, HTTPException
from app.api.request_models.query_request import QueryRequest
from app.api.response_models.query_response import QueryResponse
from app.services.query_service import QueryService

router = APIRouter()
service = QueryService()

@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        return service.handle_query(request.question, user_id=request.user_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
