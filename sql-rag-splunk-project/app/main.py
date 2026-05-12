from fastapi import FastAPI
from app.api.routes.health_routes import router as health_router
from app.api.routes.query_routes import router as query_router

app = FastAPI(
    title="SQL-RAG Splunk Project",
    description="Minimal SQL-RAG app for Splunk structured datasets",
    version="1.0.0",
)

app.include_router(health_router)
app.include_router(query_router)
