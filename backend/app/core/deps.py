"""
Shared FastAPI dependencies: current-user extraction and role gating.
Every non-public route in the API depends on `get_current_user` (or the
role-gated wrappers below).
"""
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.security import Role, decode_token, has_required_role

bearer_scheme = HTTPBearer()


class CurrentUser:
    def __init__(self, username: str, role: Role):
        self.username = username
        self.role = role


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> CurrentUser:
    token = credentials.credentials
    try:
        payload = decode_token(token)
        if payload.get("type") != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Access token required",
            )
        return CurrentUser(username=payload["sub"], role=payload["role"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )


def require_role(required_role: Role):
    """Usage: Depends(require_role("admin"))"""
    def checker(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if not has_required_role(user.role, required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role: {required_role}",
            )
        return user
    return checker
