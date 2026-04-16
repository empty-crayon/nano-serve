import pytest
from fastapi.testclient import TestClient
from main import app

TEST_MODEL = "echo" # use echo model for basic E2E without upstream GPU requirement


def test_chat_completions_echo(reset_backend):
    """Returns echo response when hitting echo profile."""
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={
            "model": TEST_MODEL,
            "messages": [{"role": "user", "content": "hello"}]
        })
        assert response.status_code == 200

        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "Your message was: hello" in data["choices"][0]["message"]["content"]


def test_chat_completions_request_id_header():
    """Response id reflects X-Request-ID header when provided."""
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": TEST_MODEL, "messages": [{"role": "user", "content": "hi"}]},
            headers={"X-Request-ID": "my-custom-id"}
        )
        assert response.status_code == 200
        assert response.json()["id"] == "my-custom-id"


def test_chat_completions_auto_request_id():
    """Generates request id automatically when X-Request-ID is missing."""
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={
            "model": TEST_MODEL,
            "messages": [{"role": "user", "content": "hi"}]
        })
        assert response.status_code == 200
        assert response.json()["id"].startswith("chatcmpl-")
