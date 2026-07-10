"""fastapi-users JWT auth: registration, login, password hashing/reset, verify, profile."""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from typing import Annotated

import structlog
from fastapi import Depends, HTTPException, Request, status
from fastapi_users import BaseUserManager, FastAPIUsers, InvalidPasswordException, UUIDIDMixin, schemas
from fastapi_users.authentication import AuthenticationBackend, BearerTransport, JWTStrategy
from fastapi_users_db_sqlalchemy import SQLAlchemyUserDatabase
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_async_session
from app.db.models import UserDB
from app.redis_client import get_redis

log = structlog.get_logger()


async def auth_throttle(request: Request) -> None:
    """Per-IP fixed-window throttle for auth endpoints (credential-stuffing guard).
    Fails open if Redis is unreachable — a cache outage must not lock users out of login."""
    s = get_settings()
    ip = request.client.host if request.client else "unknown"
    try:
        r = get_redis()
        key = f"authrl:{ip}"
        n = await r.incr(key)
        if n == 1:
            await r.expire(key, 60)
        if n > s.auth_max_per_min:
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many attempts — try again shortly.")
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — Redis down must not block authentication
        log.warning("auth_throttle.unavailable", error=str(exc))


class UserRead(schemas.BaseUser[uuid.UUID]):
    name: str = ""
    plan: str = "free"


class UserCreate(schemas.BaseUserCreate):
    name: str = ""


class UserUpdate(schemas.BaseUserUpdate):
    name: str | None = None


async def get_user_db(
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> AsyncGenerator[SQLAlchemyUserDatabase, None]:
    yield SQLAlchemyUserDatabase(session, UserDB)


class UserManager(UUIDIDMixin, BaseUserManager[UserDB, uuid.UUID]):
    def __init__(self, user_db: SQLAlchemyUserDatabase) -> None:
        super().__init__(user_db)
        secret = get_settings().jwt_secret
        self.reset_password_token_secret = secret
        self.verification_token_secret = secret

    async def validate_password(self, password: str, user) -> None:
        minimum = get_settings().min_password_length
        if len(password) < minimum:
            raise InvalidPasswordException(f"Password must be at least {minimum} characters.")
        if user.email and user.email.lower() in password.lower():
            raise InvalidPasswordException("Password must not contain your email.")

    async def on_after_register(self, user: UserDB, request: Request | None = None) -> None:
        log.info("user.registered", user_id=str(user.id), email=user.email)


async def get_user_manager(
    user_db: Annotated[SQLAlchemyUserDatabase, Depends(get_user_db)],
) -> AsyncGenerator[UserManager, None]:
    yield UserManager(user_db)


def get_jwt_strategy() -> JWTStrategy:
    s = get_settings()
    return JWTStrategy(secret=s.jwt_secret, lifetime_seconds=s.jwt_lifetime_seconds)


auth_backend = AuthenticationBackend(
    name="jwt",
    transport=BearerTransport(tokenUrl="auth/jwt/login"),
    get_strategy=get_jwt_strategy,
)

fastapi_users = FastAPIUsers[UserDB, uuid.UUID](get_user_manager, [auth_backend])
current_active_user = fastapi_users.current_user(active=True)
