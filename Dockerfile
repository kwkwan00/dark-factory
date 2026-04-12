# ── Stage 1: Build React frontend ─────────────────────────────────────────────
FROM node:20-slim AS frontend-build

WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python backend + Playwright browsers ─────────────────────────────
#
# Pinned to Debian 12 (bookworm) explicitly because Playwright's
# ``install --with-deps`` only knows about Ubuntu 20.04/22.04/24.04
# and Debian 11/12. The floating ``python:3.12-slim`` tag has rolled
# forward to Debian 13 (trixie), which Playwright does not yet
# recognise — it falls back to ubuntu20.04 package names like
# libicu66 / libvpx6 / libjpeg-turbo8 that don't exist in trixie,
# and the layer fails with "Unable to locate package". Pinning to
# bookworm keeps us on an officially supported platform until the
# upstream Playwright matrix catches up.
FROM python:3.12-slim-bookworm

WORKDIR /app

# System dependencies. `curl` is used by the HEALTHCHECK and by the
# Phase 6 E2E agent to poll server readiness before running tests.
# Node.js is required so the reconciliation + E2E deep agents can
# invoke Playwright via `npx`. We install Node via NodeSource to get
# a modern LTS on top of the python-slim Debian base.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
        | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" \
        > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Playwright browsers + system dependencies. Installed as a dedicated
# layer so the ~1 GB browser download is cached across rebuilds — it
# only re-runs when the pinned Playwright version changes. We install
# ALL THREE engines (chromium, firefox, webkit) because the Phase 6
# E2E validation stage runs smoke tests across every browser in
# ``settings.pipeline.e2e_browsers`` by default. ``--with-deps`` pulls
# in the OS libs (libnss3, libxcomposite1, libatk1.0-0, ...) that the
# browsers need to launch headlessly inside the container.
#
# We install ``@playwright/test`` GLOBALLY rather than via anonymous
# ``npx``. The ``npx --yes playwright install`` pattern works but
# emits the warning "running playwright install without first
# installing your project's dependencies" because npx downloads into
# a throwaway cache with no anchoring ``package.json``. A global
# install gives us a real, persistent project root that
# ``playwright install`` can resolve against, and also puts the
# ``playwright`` CLI on $PATH system-wide so the Phase 6 agent can
# invoke it without having to re-install the package inside every
# generated project.
#
# PLAYWRIGHT_BROWSERS_PATH=/ms-playwright is the canonical location
# used by the Microsoft Playwright base images; we match it here so
# any caller that honours that env var (including the E2E agent's
# per-run ``npx playwright install`` inside generated code) finds
# the cache already populated and skips the download.
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    PLAYWRIGHT_VERSION=1.48.0
RUN npm install -g --no-audit --no-fund \
        @playwright/test@${PLAYWRIGHT_VERSION} \
    && playwright install --with-deps chromium firefox webkit \
    && rm -rf /var/lib/apt/lists/*

# Install uv (pinned for reproducible builds)
COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies (without project)
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY src/ src/
COPY config.toml .

# Install the project itself
RUN uv sync --frozen --no-dev

# Copy React build output to the static directory served by FastAPI
COPY --from=frontend-build /app/frontend/dist src/dark_factory/api/static/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["uv", "run", "uvicorn", "dark_factory.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
