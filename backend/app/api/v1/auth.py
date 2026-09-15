"""
Auth endpoints. Login/refresh flow is fully wired with real JWT + hashing.
User lookup is a temporary in-memory store — swap for Pranjal's DB models
in Week 2 ("Auth + database + API integration").
"""
import jwt
from fastapi import APIRouter, HTTPException, Request, status

from app.core.config import settings
from app.core.limiter import limiter
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.schemas.auth import LoginRequest, RefreshRequest, TokenResponse

router = APIRouter(prefix="/auth", tags=["auth"])

# TEMP in-memory user store — replace with DB lookup in Week 2.
_FAKE_USERS_DB = {
    "analyst@example.com": {
        "username": "analyst@example.com",
        "hashed_password": hash_password("changeme123"),  # dev-only seed
        "role": "analyst",
    },
}


@router.post("/login", response_model=TokenResponse)
@limiter.limit(settings.LOGIN_RATE_LIMIT)
def login(request: Request, payload: LoginRequest):
    user = _FAKE_USERS_DB.get(payload.username)
    if not user or not verify_password(payload.password, user["hashed_password"]):
        # Intentionally vague — don't reveal whether the username exists.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    return TokenResponse(
        access_token=create_access_token(user["username"], user["role"]),
        refresh_token=create_refresh_token(user["username"], user["role"]),
    )


@router.post("/refresh", response_model=TokenResponse)
def refresh(payload: RefreshRequest):
    try:
        decoded = decode_token(payload.refresh_token)
        if decoded.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token required",
            )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
        )

    return TokenResponse(
        access_token=create_access_token(decoded["sub"], decoded["role"]),
        refresh_token=create_refresh_token(decoded["sub"], decoded["role"]),
    )
