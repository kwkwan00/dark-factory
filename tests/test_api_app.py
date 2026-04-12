"""Tests for the FastAPI application — health, routing, CORS."""

from __future__ import annotations


def test_health_endpoint(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_openapi_schema_registered(api_client):
    """All API routers are mounted under /api and appear in the OpenAPI schema."""
    resp = api_client.get("/openapi.json")
    assert resp.status_code == 200
    paths = resp.json()["paths"]
    assert "/api/agent/run" in paths
    assert "/api/graph/gaps" in paths
    assert "/api/history" in paths
    assert "/api/memory/search" in paths
    assert "/api/health" in paths
    assert "/api/watch/start" in paths
    assert "/api/watch/stop" in paths
    assert "/api/watch/status" in paths
    assert "/api/upload" in paths


def test_cors_headers_present(api_client):
    """CORS middleware allows cross-origin requests."""
    resp = api_client.options(
        "/api/health",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
    )
    assert resp.headers.get("access-control-allow-origin") in ("*", "http://localhost:3000")


def test_watcher_state_initialised(api_client):
    """App state starts with watcher=None (set in lifespan)."""
    from dark_factory.api.app import app

    assert app.state.watcher is None
