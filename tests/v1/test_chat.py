import pytest
from fastapi.testclient import TestClient
from main import app
from nanoserve.core.dependencies import get_dispatcher
from nanoserve.schemas.request_context import RequestContext

TEST_MODEL = "echo"

class MockDispatcher:
    async def dispatch(self, ctx: RequestContext) -> dict:
        return {
            "id": ctx.request_id,
            "object": "chat.completion",
            "created": 1000000000,
            "model": ctx.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": f"Echo: {ctx.messages[-1].content}"},
                "finish_reason": "stop"
            }]
        }


@pytest.fixture
def mock_client():
    app.dependency_overrides[get_dispatcher] = lambda: MockDispatcher()
    # Using 'with TestClient' ensures lifespan is triggered (needed for startup validation)
    # but since we override getter, we strictly don't even need lifespan for dispatcher,
    # except that FASTAPI 0.100+ can complain if state isn't initialized when someone checks app.state
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_chat_completions_echo(reset_backend, mock_client):
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
    response = mock_client.post(
        "/v1/chat/completions",
        json={"model": TEST_MODEL, "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Request-ID": "my-custom-id"}
    )
    assert response.status_code == 200
    assert response.json()["id"] == "my-custom-id"


def test_chat_completions_auto_request_id(mock_client):
    response = mock_client.post("/v1/chat/completions", json={
        "model": TEST_MODEL,
        "messages": [{"role": "user", "content": "hi"}]
    })
    assert response.status_code == 200
    assert response.json()["id"].startswith("chatcmpl-")


# ---- Error cases ----

def test_chat_completions_last_message_not_user(mock_client):
    response = mock_client.post("/v1/chat/completions", json={
        "model": TEST_MODEL,
        "messages": [{"role": "assistant", "content": "hello"}]
    })
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "InvalidRequestError"


def test_chat_completions_invalid_role(mock_client):
    response = mock_client.post("/v1/chat/completions", json={
        "model": TEST_MODEL,
        "messages": [{"role": "unknown", "content": "hello"}]
    })
    assert response.status_code == 422
