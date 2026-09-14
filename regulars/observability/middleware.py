"""``RequestLoggingMiddleware``: one structured log line per HTTP request, plus ``X-Request-Id``.

Never logs request or response bodies — webhook bodies carry card details and message text
(see ``regulars/app.py``'s module docstring); only method, path, status, and timing.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from regulars.observability.logging_config import bind_context
from regulars.observability.metrics import WEBHOOK_REQUESTS

logger = logging.getLogger("regulars.http")

REQUEST_ID_HEADER = "X-Request-Id"


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Binds a request id, times the request, logs one line, and records webhook metrics."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        start = time.monotonic()
        with bind_context(request_id=request_id):
            try:
                response = await call_next(request)
            except Exception:
                duration_ms = (time.monotonic() - start) * 1000
                WEBHOOK_REQUESTS.inc(route=request.url.path, status="500")
                logger.exception(
                    "http_request",
                    extra={
                        "http_method": request.method,
                        "http_path": request.url.path,
                        "http_status": 500,
                        "duration_ms": round(duration_ms, 3),
                        "request_id": request_id,
                    },
                )
                raise
            duration_ms = (time.monotonic() - start) * 1000
            response.headers[REQUEST_ID_HEADER] = request_id
            WEBHOOK_REQUESTS.inc(route=request.url.path, status=str(response.status_code))
            logger.info(
                "http_request",
                extra={
                    "http_method": request.method,
                    "http_path": request.url.path,
                    "http_status": response.status_code,
                    "duration_ms": round(duration_ms, 3),
                    "request_id": request_id,
                },
            )
            return response
