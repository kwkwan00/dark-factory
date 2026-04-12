"""Tests for the model-list proxy endpoints.

Covers:
- ``POST /api/models/anthropic``
- ``POST /api/models/openai``

Both endpoints share the same response shape and fallback behaviour,
so the tests exercise:
- No API key in body or env → ``source="fallback"`` + hardcoded list
- Happy path with mocked upstream → ``source="live"`` + parsed list
- Upstream 401 → endpoint returns 401 with an error message
- Upstream network error → endpoint returns 503
- OpenAI filter correctly excludes embeddings / whisper / tts / etc.
- API key never logged (defence against accidental secret leakage)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


# ── /api/models/anthropic ──────────────────────────────────────────────────


def test_anthropic_models_no_key_returns_fallback(api_client, monkeypatch):
    """When neither body nor env provides a key, the endpoint degrades
    to the hardcoded fallback list with ``source="fallback"``."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    resp = api_client.post("/api/models/anthropic", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "fallback"
    assert len(body["models"]) > 0
    # Contains the known fallback ids
    ids = {m["id"] for m in body["models"]}
    assert "claude-opus-4-6" in ids


def test_anthropic_models_empty_string_key_treated_as_none(api_client, monkeypatch):
    """Empty-string keys from the frontend (user typed then cleared)
    must be normalised to ``None`` so we fall back cleanly instead of
    hitting the API with an empty credential."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    resp = api_client.post("/api/models/anthropic", json={"api_key": ""})
    assert resp.status_code == 200
    assert resp.json()["source"] == "fallback"


def test_anthropic_models_whitespace_key_rejected(api_client):
    """Whitespace inside a key is a copy-paste mistake, not a valid key."""
    resp = api_client.post(
        "/api/models/anthropic", json={"api_key": "sk-ant-bad key"}
    )
    assert resp.status_code == 422


def test_anthropic_models_oversized_key_rejected(api_client):
    """Giant blobs are almost certainly pasted by mistake. Cap at 512."""
    resp = api_client.post(
        "/api/models/anthropic", json={"api_key": "sk-" + "x" * 600}
    )
    assert resp.status_code == 422


def test_anthropic_models_happy_path(api_client):
    """Mock the upstream /v1/models response and verify the endpoint
    parses + sorts it correctly. Newest model should come first."""

    class _FakeResp:
        status_code = 200

        def json(self):
            return {
                "data": [
                    {
                        "id": "claude-sonnet-4-6",
                        "display_name": "Claude Sonnet 4.6",
                        "created_at": "2025-03-01T00:00:00Z",
                    },
                    {
                        "id": "claude-opus-4-6",
                        "display_name": "Claude Opus 4.6",
                        "created_at": "2025-06-01T00:00:00Z",
                    },
                ],
                "has_more": False,
            }

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            self.calls: list = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, headers=None, params=None):
            self.calls.append((url, headers, params))
            return _FakeResp()

    with patch(
        "dark_factory.api.routes_models.httpx.Client",
        new=lambda *a, **kw: _FakeClient(),
    ):
        resp = api_client.post(
            "/api/models/anthropic",
            json={"api_key": "sk-ant-valid-test-key"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "live"
    ids = [m["id"] for m in body["models"]]
    # Newest first — opus-4-6 (2025-06-01) before sonnet-4-6 (2025-03-01)
    assert ids == ["claude-opus-4-6", "claude-sonnet-4-6"]


def test_anthropic_models_401_from_upstream(api_client):
    """Invalid API key → 401 from Anthropic → endpoint returns 401
    with a human-readable message the frontend can surface."""

    class _FakeResp:
        status_code = 401
        text = "invalid_api_key"

        def json(self):
            return {"error": "invalid_api_key"}

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, *a, **kw):
            return _FakeResp()

    with patch(
        "dark_factory.api.routes_models.httpx.Client",
        new=lambda *a, **kw: _FakeClient(),
    ):
        resp = api_client.post(
            "/api/models/anthropic", json={"api_key": "sk-ant-bad"}
        )
    assert resp.status_code == 401
    assert "invalid" in resp.json()["detail"].lower()


def test_anthropic_models_network_error_returns_503(api_client):
    """Network failures surface as 503 with the httpx error attached."""
    import httpx as _httpx

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, *a, **kw):
            raise _httpx.ConnectError("connection refused")

    with patch(
        "dark_factory.api.routes_models.httpx.Client",
        new=lambda *a, **kw: _FakeClient(),
    ):
        resp = api_client.post(
            "/api/models/anthropic", json={"api_key": "sk-ant-valid"}
        )
    assert resp.status_code == 503
    assert "connection refused" in resp.json()["detail"].lower()


def test_anthropic_models_uses_env_var_when_body_empty(api_client, monkeypatch):
    """With no body key but ``ANTHROPIC_API_KEY`` set, the endpoint must
    still attempt a live fetch (not fall back)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env-key")

    class _FakeResp:
        status_code = 200

        def json(self):
            return {"data": [{"id": "claude-haiku-env", "display_name": "H"}], "has_more": False}

    captured_headers: dict = {}

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, headers=None, params=None):
            captured_headers.update(headers or {})
            return _FakeResp()

    with patch(
        "dark_factory.api.routes_models.httpx.Client",
        new=lambda *a, **kw: _FakeClient(),
    ):
        resp = api_client.post("/api/models/anthropic", json={})

    assert resp.status_code == 200
    assert resp.json()["source"] == "live"
    # The env var value was forwarded in the x-api-key header
    assert captured_headers.get("x-api-key") == "sk-ant-env-key"


# ── /api/models/openai ─────────────────────────────────────────────────────


def test_openai_models_no_key_returns_fallback(api_client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    resp = api_client.post("/api/models/openai", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "fallback"
    ids = {m["id"] for m in body["models"]}
    assert "gpt-5.4" in ids or "gpt-4.1" in ids


def test_openai_models_filter_excludes_non_chat(api_client):
    """The upstream /v1/models dump includes embeddings, whisper, dall-e,
    moderation, and tts alongside chat models. The endpoint must filter
    down to chat/reasoning models only so DeepEval doesn't see anything
    it can't use as a judge."""

    class _FakeResp:
        status_code = 200

        def json(self):
            return {
                "data": [
                    {"id": "gpt-4.1", "created": 1700000000},
                    {"id": "gpt-4.1-mini", "created": 1700000001},
                    {"id": "gpt-4o", "created": 1700000002},
                    {"id": "o4-mini", "created": 1700000003},
                    {"id": "chatgpt-4o-latest", "created": 1700000004},
                    # These must be filtered OUT:
                    {"id": "text-embedding-3-large", "created": 1700000000},
                    {"id": "whisper-1", "created": 1700000000},
                    {"id": "tts-1-hd", "created": 1700000000},
                    {"id": "dall-e-3", "created": 1700000000},
                    {"id": "omni-moderation-latest", "created": 1700000000},
                    {"id": "gpt-3.5-turbo-instruct", "created": 1700000000},
                    {"id": "text-davinci-003", "created": 1700000000},
                ]
            }

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, *a, **kw):
            return _FakeResp()

    with patch(
        "dark_factory.api.routes_models.httpx.Client",
        new=lambda *a, **kw: _FakeClient(),
    ):
        resp = api_client.post(
            "/api/models/openai", json={"api_key": "sk-test-valid"}
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "live"
    ids = [m["id"] for m in body["models"]]
    # All five chat models kept, in newest-first order
    assert ids == [
        "chatgpt-4o-latest",
        "o4-mini",
        "gpt-4o",
        "gpt-4.1-mini",
        "gpt-4.1",
    ]
    # None of the excluded types survived
    for bad in (
        "text-embedding-3-large",
        "whisper-1",
        "tts-1-hd",
        "dall-e-3",
        "omni-moderation-latest",
        "gpt-3.5-turbo-instruct",
        "text-davinci-003",
    ):
        assert bad not in ids, f"{bad} should have been filtered"


def test_openai_models_401_from_upstream(api_client):
    class _FakeResp:
        status_code = 401
        text = "invalid"

        def json(self):
            return {"error": {"message": "Invalid Authentication"}}

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, *a, **kw):
            return _FakeResp()

    with patch(
        "dark_factory.api.routes_models.httpx.Client",
        new=lambda *a, **kw: _FakeClient(),
    ):
        resp = api_client.post(
            "/api/models/openai", json={"api_key": "sk-bad"}
        )
    assert resp.status_code == 401
    assert "invalid" in resp.json()["detail"].lower()


def test_openai_models_unexpected_status_returns_502(api_client):
    """A 500 from OpenAI surfaces as 502 (bad gateway) so the frontend
    can distinguish "our upstream broke" from "your key is wrong"."""

    class _FakeResp:
        status_code = 500
        text = "internal_server_error"

        def json(self):
            return {}

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, *a, **kw):
            return _FakeResp()

    with patch(
        "dark_factory.api.routes_models.httpx.Client",
        new=lambda *a, **kw: _FakeClient(),
    ):
        resp = api_client.post(
            "/api/models/openai", json={"api_key": "sk-valid"}
        )
    assert resp.status_code == 502


# ── Key safety ─────────────────────────────────────────────────────────────


def test_api_key_is_never_logged(api_client, caplog):
    """The api_key field must NEVER appear in structlog output, even
    on happy paths. Regression guard against future log.info calls
    that accidentally pass kwargs through."""
    import logging

    caplog.set_level(logging.DEBUG)

    class _FakeResp:
        status_code = 200

        def json(self):
            return {"data": [], "has_more": False}

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, *a, **kw):
            return _FakeResp()

    secret = "sk-ant-super-secret-do-not-leak"
    with patch(
        "dark_factory.api.routes_models.httpx.Client",
        new=lambda *a, **kw: _FakeClient(),
    ):
        resp = api_client.post(
            "/api/models/anthropic", json={"api_key": secret}
        )
    assert resp.status_code == 200

    # The secret MUST NOT appear anywhere in captured log output
    for record in caplog.records:
        assert secret not in record.getMessage(), (
            f"API key leaked into log record: {record.getMessage()}"
        )
