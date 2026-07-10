"""Shared test helpers: asset paths and skip markers for real-infra integration tests."""

from __future__ import annotations

import socket
from pathlib import Path
from urllib.parse import urlparse

import pytest

from app.config import get_settings

ASSETS = Path(__file__).parent / "assets"


def _qdrant_up() -> bool:
    u = urlparse(get_settings().qdrant_url)
    try:
        with socket.create_connection((u.hostname or "localhost", u.port or 6333), timeout=1):
            return True
    except OSError:
        return False


needs_gemini = pytest.mark.skipif(not get_settings().gemini_api_key, reason="no GEMINI_API_KEY")
needs_qdrant = pytest.mark.skipif(not _qdrant_up(), reason="Qdrant not reachable")
