import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import Depends, FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from app.api.chats import router as chats_router
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.config import get_settings
from app.db import create_all, get_engine
from app.llm import aclose_llm
from app.logging import configure_logging
from app.rag.vectorstore import aclose_vector_store, get_vector_store
from app.redis_client import aclose_redis
from app.users import UserCreate, UserRead, UserUpdate, auth_backend, auth_throttle, fastapi_users

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    configure_logging(s.log_level, json_logs=(s.app_env == "prod"))
    log.info("app.startup", env=s.app_env)

    if s.app_env == "prod" and s.jwt_secret == "CHANGE_ME_IN_PROD":
        raise RuntimeError("JWT_SECRET must be overridden in prod")
    if not s.gemini_api_key:
        log.warning("llm.no_api_key", hint="set GEMINI_API_KEY — chat and embeddings will fail without it")

    await create_all()
    get_vector_store()  # construct the shared client (no network until first use)

    app.state.arq_pool = None
    if s.ingest_executor == "queue":
        try:
            from arq import create_pool
            from arq.connections import RedisSettings

            app.state.arq_pool = await create_pool(RedisSettings.from_dsn(s.redis_url))
            log.info("arq.pool_ready")
        except Exception as exc:  # noqa: BLE001 — fall back to inline ingestion
            log.error("arq.pool_unavailable", error=str(exc), hint="ingestion falls back to inline")

    try:
        yield
    finally:
        log.info("app.shutdown")
        if app.state.arq_pool is not None:
            await app.state.arq_pool.aclose()
        await aclose_vector_store()
        await aclose_llm()
        await aclose_redis()
        await get_engine().dispose()


async def _request_logger(request: Request, call_next):
    """Bind a correlation id, cap non-multipart body size, log the request lifecycle."""
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    structlog.contextvars.bind_contextvars(request_id=rid)

    s = get_settings()
    ctype = request.headers.get("content-type", "")
    if not ctype.startswith("multipart/form-data"):
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > s.max_json_body_kb * 1024:
            structlog.contextvars.clear_contextvars()
            return JSONResponse({"detail": "request body too large"}, status.HTTP_413_CONTENT_TOO_LARGE)

    start = time.monotonic()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("request.unhandled", method=request.method, path=request.url.path)
        structlog.contextvars.clear_contextvars()
        return JSONResponse(
            {"detail": "Internal server error"},
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            headers={"X-Request-ID": rid},
        )
    log.info(
        "request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=int((time.monotonic() - start) * 1000),
    )
    structlog.contextvars.clear_contextvars()
    response.headers["X-Request-ID"] = rid
    return response


async def _security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    if get_settings().app_env == "prod":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


_settings = get_settings()
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["120/minute"],
    storage_uri=_settings.redis_url if _settings.app_env == "prod" else None,
)


def create_app() -> FastAPI:
    app = FastAPI(title="DocuMind — Document RAG Assistant", version="0.1.0", lifespan=lifespan)

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in _settings.cors_origins.split(",") if o.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.middleware("http")(_security_headers)
    app.middleware("http")(_request_logger)

    throttle = [Depends(auth_throttle)]
    app.include_router(health_router)
    app.include_router(
        fastapi_users.get_auth_router(auth_backend), prefix="/auth/jwt", tags=["auth"], dependencies=throttle
    )
    app.include_router(
        fastapi_users.get_register_router(UserRead, UserCreate),
        prefix="/auth",
        tags=["auth"],
        dependencies=throttle,
    )
    app.include_router(fastapi_users.get_reset_password_router(), prefix="/auth", tags=["auth"])
    app.include_router(fastapi_users.get_verify_router(UserRead), prefix="/auth", tags=["auth"])
    app.include_router(fastapi_users.get_users_router(UserRead, UserUpdate), prefix="/users", tags=["users"])
    app.include_router(documents_router)
    app.include_router(chats_router)

    Instrumentator().instrument(app).expose(app)

    static_dir = Path(__file__).resolve().parent.parent / "static"
    if static_dir.is_dir():
        from starlette.responses import FileResponse

        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa(full_path: str):
            target = (static_dir / full_path).resolve()
            if target.is_relative_to(static_dir.resolve()) and target.is_file():
                return FileResponse(target)
            return FileResponse(static_dir / "index.html")

    return app


app = create_app()
