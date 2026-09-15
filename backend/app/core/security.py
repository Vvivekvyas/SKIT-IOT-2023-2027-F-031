"""
Security utilities for the Hybrid IDS API.

Covers:
- password hashing (argon2)
- JWT access + refresh token creation/verification
- role-based scope checks

This is the concrete implementation of Week 1's "security requirements":
short-lived access tokens, rotating refresh tokens, no plaintext secrets.
"""
from datetime import datetime, timedelta, timezone
from typing import Literal

import jwt
from passlib.context import CryptContext

from app.core.config import settings

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

Role = Literal["viewer", "analyst", "admin"]


# ---- Password hashing ----

def hash_password(plain_password: str) -> str:
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


# ---- JWT tokens ----

def _create_token(subject: str, role: Role, token_type: str, expires_delta: timedelta) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "role": role,
        "type": token_type,  # "access" | "refresh"
        "iat": now,
        "exp": now + expires_delta,
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_access_token(subject: str, role: Role) -> str:
    return _create_token(
        subject, role, "access",
        timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )


def create_refresh_token(subject: str, role: Role) -> str:
    return _create_token(
        subject, role, "refresh",
        timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
    )


def decode_token(token: str) -> dict:
    """Raises jwt.InvalidTokenError (or subclasses) on bad/expired tokens."""
    return jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])


# ---- Role checks ----

ROLE_RANK = {"viewer": 0, "analyst": 1, "admin": 2}


def has_required_role(user_role: Role, required_role: Role) -> bool:
    return ROLE_RANK.get(user_role, -1) >= ROLE_RANK.get(required_role, 99)
