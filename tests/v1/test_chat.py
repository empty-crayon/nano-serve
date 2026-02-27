from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_chat_completions_echo():
    """Returns echo response when backend_url is not set."""
    response = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}]
    })
    assert response.status_code == 200

    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert "Echo: hello" in data["choices"][0]["message"]["content"]


def test_chat_completions_request_id_header():
    """Response id reflects X-Request-ID header when provided."""
    response = client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Request-ID": "my-custom-id"}
    )
    assert response.status_code == 200
    assert response.json()["id"] == "my-custom-id"


def test_chat_completions_auto_request_id():
    """Generates request id automatically when X-Request-ID is missing."""
    response = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}]
    })
    assert response.status_code == 200
    assert response.json()["id"].startswith("chatcmpl-")

# ---- Error cases ----

def test_chat_completions_last_message_not_user():
    """Returns 400 when last message does not have user role."""
    response = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "assistant", "content": "hello"}]
    })
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "InvalidRequestError"


def test_chat_completions_invalid_role():
    """Returns 422 for undefined role (Pydantic validation)."""
    response = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "unknown", "content": "hello"}]
    })
    assert response.status_code == 422
