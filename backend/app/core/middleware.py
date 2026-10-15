"""
Request ID + access logging.

Every request gets an ID (returned in the X-Request-ID response header and
in error bodies), so one failing request can be traced in the server log.
Only the method, path, status and time are logged. Never bodies, passwords
or tokens.
"""
import logging
import re
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger("ids.api")

# Only accept a client-supplied ID if it is short and plain, so it cannot
# be used to inject fake lines into the logs.
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9\-]{1,64}$")


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        incoming = request.headers.get("X-Request-ID", "")
        request_id = incoming if _SAFE_REQUEST_ID.match(incoming) else str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        response.headers["X-Request-ID"] = request_id
        logger.info(
            "%s %s -> %s (%.0f ms) request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request_id,
        )
        return response
