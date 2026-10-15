"""
Central error handling.

Every error the API returns has the same shape:

    {
      "data": null,
      "meta": {"request_id": "..."},
      "error": {"code": "not_found", "message": "Dataset not found", "details": null}
    }

- 4xx errors show a clear message the client can display.
- Validation errors (422) list each bad field. The submitted value is never
  echoed back, so a wrong password is not repeated in the response.
- Unexpected errors (500) return a generic message. The real traceback only
  goes to the server log, together with the request_id, so it can be found.
"""
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("ids.api")

ERROR_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    422: "validation_error",
    429: "rate_limited",
    500: "internal_error",
}


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def error_response(
    request: Request,
    status_code: int,
    message: str,
    details: list[dict] | None = None,
    headers: dict | None = None,
    code: str | None = None,
) -> JSONResponse:
    request_id = _request_id(request)
    out_headers = dict(headers or {})
    if request_id:
        out_headers["X-Request-ID"] = request_id
    return JSONResponse(
        status_code=status_code,
        headers=out_headers,
        content={
            "data": None,
            "meta": {"request_id": request_id},
            "error": {
                "code": code or ERROR_CODES.get(status_code, f"http_{status_code}"),
                "message": message,
                "details": details,
            },
        },
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    message = exc.detail if isinstance(exc.detail, str) else "Request failed"
    return error_response(request, exc.status_code, message, headers=exc.headers)


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    details = [
        {
            "field": ".".join(str(part) for part in err["loc"]),
            "message": err["msg"],
        }
        for err in exc.errors()
    ]
    return error_response(request, 422, "Request validation failed", details=details)


async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return error_response(request, 429, "Too many requests. Please try again later.")


async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception(
        "Unhandled error request_id=%s %s %s",
        _request_id(request),
        request.method,
        request.url.path,
    )
    return error_response(request, 500, "Something went wrong on our side.")


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
