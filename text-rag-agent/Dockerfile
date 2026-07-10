# Combined image: build the SPA, then serve it from the FastAPI app's static mount.
# Build context is the repo root (needs both frontend/ and backend/).

FROM node:20-alpine AS frontend
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# antiword: legacy .doc extraction (the only non-Python runtime dep)
RUN apt-get update && apt-get install -y --no-install-recommends antiword \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev

COPY backend/app ./app
COPY --from=frontend /fe/dist ./static

EXPOSE 8000
# Shell form so ${PORT} (injected by Railway/Render/Fly/etc.) expands; defaults to 8000 locally.
CMD uv run --no-dev uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
