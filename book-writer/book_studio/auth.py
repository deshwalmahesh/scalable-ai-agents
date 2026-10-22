"""Password hashing and bearer authentication shared by the HTTP endpoints."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer
from jwt.exceptions import InvalidTokenError
from pwdlib import PasswordHash


PASSWORD_HASH = PasswordHash.recommended()
DUMMY_HASH = PASSWORD_HASH.hash("missing-user-password")
BEARER = OAuth2PasswordBearer(tokenUrl="/auth/token")


def current_user(request: Request, token: Annotated[str, Depends(BEARER)]) -> dict:
    unauthorized = HTTPException(status_code=401, detail="Invalid or expired credentials",
                                 headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, request.app.state.jwt_secret, algorithms=["HS256"],
                             options={"require": ["sub", "exp"]})
        user_id = int(payload["sub"])
    except (InvalidTokenError, ValueError, TypeError, KeyError) as exc:
        raise unauthorized from exc
    user = request.app.state.service.store.user_by_id(user_id)
    if user is None:
        raise unauthorized
    return user


def access_token(user_id: int, secret: str) -> str:
    expires = datetime.now(timezone.utc) + timedelta(hours=1)
    return jwt.encode({"sub": str(user_id), "exp": expires}, secret, algorithm="HS256")
