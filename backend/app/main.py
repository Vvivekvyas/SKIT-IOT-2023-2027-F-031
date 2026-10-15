"""
App entrypoint. Wires CORS, rate limiting, request IDs, error handling
and versioned routers.
Run with: uvicorn app.main:app --reload
Before first run: python -m app.db.seed
"""
import logging

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import auth, datasets
from app.core.config import settings
from app.core.deps import CurrentUser, get_current_user
from app.core.errors import register_error_handlers
from app.core.limiter import limiter
from app.core.middleware import RequestIDMiddleware

logging.basicConfig(level=logging.INFO)

app = FastAPI(title=settings.APP_NAME)
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


@app.get("/health")
def health():
    """Public — no auth required."""
    return {"status": "ok"}


@app.get(f"{settings.API_V1_PREFIX}/me")
def read_current_user(user: CurrentUser = Depends(get_current_user)):
    return {"username": user.username, "role": user.role}
