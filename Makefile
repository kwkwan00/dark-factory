# ═══════════════════════════════════════════════════════════════════════════
#  AI Dark Factory — build + run orchestration
# ═══════════════════════════════════════════════════════════════════════════
#
# Typical flows:
#
#   make build      # Build the dark-factory app image via docker compose.
#                   # The Dockerfile installs Python 3.12 (bookworm), Node
#                   # 20, uv, and Playwright browsers inline — first build
#                   # takes ~5–10 min for the browser download, subsequent
#                   # builds are fast because Docker caches that layer
#                   # until the Playwright version pin changes.
#
#   make up         # docker compose up -d (starts the full stack).
#
#   make logs       # Tail dark-factory logs.
#
#   make down       # Stop and remove containers.
#
#   make test       # Run the Python test suite.
#
# ═══════════════════════════════════════════════════════════════════════════

.PHONY: help build up down restart logs ps test tsc fmt lint clean

help:
	@echo "Targets:"
	@echo "  build      Build the dark-factory app image"
	@echo "  up         Start the full stack (docker compose up -d)"
	@echo "  down       Stop and remove containers"
	@echo "  restart    Down + up"
	@echo "  logs       Tail dark-factory logs"
	@echo "  ps         Show compose service status"
	@echo "  test       Run the Python test suite (uv run pytest)"
	@echo "  tsc        Run the frontend TypeScript check"
	@echo "  fmt        Run ruff format"
	@echo "  lint       Run ruff check"
	@echo "  clean      Remove local build cache + frontend dist"

# ── Build + compose lifecycle ────────────────────────────────────────────────

build:
	docker compose build dark-factory

up:
	docker compose up -d

down:
	docker compose down

restart: down up

logs:
	docker compose logs -f dark-factory

ps:
	docker compose ps

# ── Local dev ────────────────────────────────────────────────────────────────

test:
	uv run pytest tests/ -v

tsc:
	cd frontend && npx tsc --noEmit

fmt:
	uv run ruff format src/ tests/

lint:
	uv run ruff check src/ tests/

clean:
	rm -rf frontend/dist frontend/node_modules/.vite
	rm -rf .pytest_cache .ruff_cache
	rm -rf src/dark_factory/api/static
