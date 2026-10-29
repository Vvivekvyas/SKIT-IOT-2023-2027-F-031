"""
App entrypoint. Wires CORS, rate limiting, request IDs, error handling
and versioned routers.
Run with: uvicorn app.main:app --reload
Before first run: python -m app.db.seed
"""
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import auth, datasets, predict
from app.core.config import settings
from app.core.deps import CurrentUser, get_current_user
from app.core.errors import register_error_handlers
from app.core.limiter import limiter
from app.core.middleware import RequestIDMiddleware
from app.db.schema import ensure_schema
from app.services.predictor import prediction_service

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Make sure tables and newer columns exist before the first request.
    ensure_schema()
    # Load the prediction model once. Never raises; if it fails, /predict answers 503.
    prediction_service.load()
    yield


app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)
app.state.limiter = limiter
register_error_handlers(app)

app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
)

app.include_router(auth.router, prefix=settings.API_V1_PREFIX)
app.include_router(datasets.router, prefix=settings.API_V1_PREFIX)
app.include_router(predict.router, prefix=settings.API_V1_PREFIX)


@app.get("/health")
def health():
    """Public - no auth required."""
    return {"status": "ok"}


@app.get(f"{settings.API_V1_PREFIX}/me")
def read_current_user(user: CurrentUser = Depends(get_current_user)):
    return {"username": user.username, "role": user.role}
