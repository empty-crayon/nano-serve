import pytest
from fastapi.testclient import TestClient
from main import app
from nanorelay.schemas.chat import ChatMessage
from nanorelay.core.dependencies import get_dispatcher


TEST_MODEL = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"


class MockDispatcher:
    async def dispatch(self, messages: list[ChatMessage], model: str) -> dict:
        return {
            "id": "backend-generated-id",
            "object": "chat.completion",
            "created": 1000000000,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": f"Echo: {messages[-1].content}"},
                "finish_reason": "stop"
            }]
        }


@pytest.fixture
def mock_client():
    app.dependency_overrides[get_dispatcher] = lambda: MockDispatcher()
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_chat_completions_echo(reset_backend, mock_client):
    """Returns echo response when backend_url is not set."""

    response = mock_client.post("/v1/chat/completions", json={
        "model": TEST_MODEL,
        "messages": [{"role": "user", "content": "hello"}]
    })
    assert response.status_code == 200

    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert "Echo: hello" in data["choices"][0]["message"]["content"]


def test_chat_completions_request_id_header(mock_client):
    """Response id reflects X-Request-ID header when provided."""
    response = mock_client.post(
        "/v1/chat/completions",
        json={"model": TEST_MODEL, "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Request-ID": "my-custom-id"}
    )
    assert response.status_code == 200
    assert response.json()["id"] == "my-custom-id"


def test_chat_completions_auto_request_id(mock_client):
    """Generates request id automatically when X-Request-ID is missing."""
    response = mock_client.post("/v1/chat/completions", json={
        "model": TEST_MODEL,
        "messages": [{"role": "user", "content": "hi"}]
    })
    assert response.status_code == 200
    assert response.json()["id"].startswith("chatcmpl-")

# ---- Error cases ----

def test_chat_completions_last_message_not_user(mock_client):
    """Returns 400 when last message does not have user role."""
    response = mock_client.post("/v1/chat/completions", json={
        "model": TEST_MODEL,
        "messages": [{"role": "assistant", "content": "hello"}]
    })
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "InvalidRequestError"


def test_chat_completions_invalid_role(mock_client):
    """Returns 422 for undefined role (Pydantic validation)."""
    response = mock_client.post("/v1/chat/completions", json={
        "model": TEST_MODEL,
        "messages": [{"role": "unknown", "content": "hello"}]
    })
    assert response.status_code == 422
