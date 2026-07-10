"""Test fixtures. Env is set BEFORE any app import so get_settings() caches test values.

Integration tests use REAL Qdrant + REAL Gemini (no mocking of the RAG pipeline); they
skip automatically when GEMINI_API_KEY or a reachable Qdrant is absent.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

# --- set test env before importing app ---------------------------------------
_TEST_DB = Path(__file__).parent / "_test.db"
os.environ.setdefault("APP_ENV", "test")
# Default SQLite; set TEST_DATABASE_URL to run the same suite against Postgres.
os.environ["DATABASE_URL"] = os.environ.get("TEST_DATABASE_URL", f"sqlite+aiosqlite:///{_TEST_DB}")
os.environ["INGEST_EXECUTOR"] = "inline"
os.environ["QDRANT_COLLECTION"] = f"test_{uuid.uuid4().hex[:8]}"
os.environ["AUTH_MAX_PER_MIN"] = "100000"  # don't throttle the many logins tests make

# Load GEMINI_API_KEY from the repo .env if present (so real E2E can run locally).
_root_env = Path(__file__).resolve().parents[2] / ".env"
if _root_env.is_file() and not os.environ.get("GEMINI_API_KEY"):
    for line in _root_env.read_text().splitlines():
        if line.startswith("GEMINI_API_KEY=") and "=" in line:
            os.environ["GEMINI_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")

import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from asgi_lifespan import LifespanManager  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _cleanup_db():
    yield
    _TEST_DB.unlink(missing_ok=True)


@pytest_asyncio.fixture
async def client():
    from app.main import app

    async with (
        LifespanManager(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c,
    ):
        yield c


@pytest_asyncio.fixture
async def auth_headers(client):
    email = f"user_{uuid.uuid4().hex[:8]}@test.dev"
    password = "s3cure-password"
    await client.post("/auth/register", json={"email": email, "password": password})
    resp = await client.post("/auth/jwt/login", data={"username": email, "password": password})
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
