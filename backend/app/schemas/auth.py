from pydantic import BaseModel, ConfigDict, EmailStr, Field


class LoginRequest(BaseModel):
    # extra="forbid": unknown fields are rejected instead of silently ignored
    model_config = ConfigDict(extra="forbid")

    username: EmailStr = Field(max_length=254)
    # max_length stops someone sending a huge password to slow down hashing
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    refresh_token: str = Field(min_length=1, max_length=2048)
