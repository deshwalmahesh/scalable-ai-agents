.DEFAULT_GOAL := help
.PHONY: help up down logs test clean

help:  ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-8s\033[0m %s\n", $$1, $$2}'

up:  ## Build + run the whole stack (Postgres, Redis, Qdrant, API+UI, worker) at http://localhost:8000
	@test -f .env || { \
	  echo "→ creating .env (set GEMINI_API_KEY in your shell to fill it automatically)"; \
	  printf 'GEMINI_API_KEY=%s\nJWT_SECRET=%s\n' "$${GEMINI_API_KEY:-}" \
	    "$$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')" > .env; \
	}
	docker compose up -d --build
	@echo "→ DocuMind is up: http://localhost:8000  (Swagger /docs) · 'make logs' to watch, 'make down' to stop"

down:  ## Stop the stack (keeps data volumes)
	docker compose down

logs:  ## Tail logs from all services
	docker compose logs -f

test:  ## Run unit + smoke tests (no external services needed)
	cd backend && uv run pytest tests/unit tests/smoke -q

clean:  ## Stop the stack AND delete data volumes (Postgres/Redis/Qdrant)
	docker compose down -v
