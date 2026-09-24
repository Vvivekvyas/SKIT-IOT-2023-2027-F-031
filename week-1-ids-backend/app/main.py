"""
Application entrypoint.

Run locally with:
    uvicorn app.main:app --reload
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import api_router, pages_router
from app.core.config import settings
from app.core.database import Base, engine

# Create tables on startup for local dev. In production, switch to Alembic
# migrations instead of relying on create_all().
Base.metadata.create_all(bind=engine)

app = FastAPI(title=settings.PROJECT_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

# JSON API, e.g. /api/v1/predict, /api/v1/traffic/records
app.include_router(api_router, prefix=settings.API_V1_PREFIX)

# Server-rendered pages, e.g. /login, /dashboard
app.include_router(pages_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
