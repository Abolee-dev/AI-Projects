from fastapi import APIRouter, Depends

from container import get_rag_chain
from llm.rag_chain import RAGChain
from models.schemas import QueryRequest, QueryResponse

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("/", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    chain: RAGChain = Depends(get_rag_chain),
) -> QueryResponse:
    return await chain.run(request)
