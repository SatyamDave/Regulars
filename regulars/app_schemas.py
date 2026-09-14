"""Response models for ``regulars.app``'s OpenAPI schema.

These exist purely to give the auto-generated schema (and the ``/docs`` Swagger UI) concrete
shapes to show. They document response bodies that already exist; they do not change what any
route actually returns. Webhook routes return a bare ``fastapi.Response`` with an empty body, so
FastAPI never serializes ``WebhookAckResponse`` against them at runtime — it is documentation of
the conceptual "yes, this was accepted" contract, not a body FastAPI emits.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Body returned by ``GET /healthz``."""

    status: Literal["ok"] = Field(description='Always "ok" when the process can serve requests.')


class WebhookAckResponse(BaseModel):
    """Conceptual acknowledgement shape for a successfully handled webhook.

    Documentation only: both webhook routes actually return an empty ``200``/``204``-style
    ``Response`` with no body, so this model never gets serialized at runtime. It is attached as
    ``response_model`` so the OpenAPI schema and Swagger UI show integrators what "success" means.
    """

    received: bool = Field(description="True once the payload has been handed to its sink.")


class ErrorResponse(BaseModel):
    """Shape of the JSON body FastAPI's default handler renders for an ``HTTPException``."""

    detail: str = Field(description="Human-readable reason the request was rejected.")
