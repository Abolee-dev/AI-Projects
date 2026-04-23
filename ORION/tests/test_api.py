import pytest
from httpx import AsyncClient, ASGITransport

from api.main import app


@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_query_validation():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # empty query should fail validation
        resp = await client.post("/query/", json={"query": ""})
    assert resp.status_code == 422
