"""
App entrypoint. Wires CORS, rate limiting, and versioned routers.
Run with: uvicorn app.main:app --reload
Before first run: python -m app.db.seed
"""
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.v1 import auth
from app.core.config import settings
from app.core.deps import CurrentUser, get_current_user
from app.core.limiter import limiter

app = FastAPI(title=settings.APP_NAME)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(auth.router, prefix=settings.API_V1_PREFIX)


@app.get("/health")
def health():
    """Public — no auth required."""
    return {"status": "ok"}


@app.get(f"{settings.API_V1_PREFIX}/me")
def read_current_user(user: CurrentUser = Depends(get_current_user)):
    return {"username": user.username, "role": user.role}
