"""Failure taxonomy: map any exception to a stable code, user message, and retry policy.

Single source of truth so every surface (ingestion worker, query stream, API handlers)
reports the same thing for the same failure. Classification inspects HTTP status and
exception class name rather than importing provider SDKs, so it is provider-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

INTERNAL = "INTERNAL"
INPUT_INVALID = "INPUT_INVALID"
AUTH = "AUTH"
RATE_LIMITED = "RATE_LIMITED"
PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"


@dataclass(frozen=True)
class ErrorInfo:
    code: str
    user_message: str


_INFO: dict[str, str] = {
    INTERNAL: "Something went wrong on our side.",
    INPUT_INVALID: "The request could not be processed — try rephrasing or shortening it.",
    AUTH: "Model credentials are invalid or expired.",
    RATE_LIMITED: "The model is busy right now — please try again shortly.",
    PROVIDER_UNAVAILABLE: "The model provider is temporarily unavailable.",
}


def info_for(code: str) -> ErrorInfo:
    return ErrorInfo(code=code, user_message=_INFO.get(code, _INFO[INTERNAL]))


def _status_code(exc: BaseException) -> int | None:
    val = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    return val if isinstance(val, int) else None


def classify(exc: BaseException) -> ErrorInfo:
    name = type(exc).__name__.lower()
    status = _status_code(exc)
    if status is not None:
        if status in (401, 403):
            return info_for(AUTH)
        if status == 429:
            return info_for(RATE_LIMITED)
        if status in (408, 409, 425) or status >= 500:
            return info_for(PROVIDER_UNAVAILABLE)
        if 400 <= status < 500:
            return info_for(INPUT_INVALID)
    if "ratelimit" in name:
        return info_for(RATE_LIMITED)
    if any(k in name for k in ("authentic", "permission", "apikey", "unauthorized")):
        return info_for(AUTH)
    if any(k in name for k in ("badrequest", "invalidrequest", "unprocessable", "validation", "notfound")):
        return info_for(INPUT_INVALID)
    if any(k in name for k in ("timeout", "connection", "overloaded", "serviceunavailable", "apierror")):
        return info_for(PROVIDER_UNAVAILABLE)
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return info_for(PROVIDER_UNAVAILABLE)
    return info_for(INTERNAL)
