import pytest


@pytest.fixture
def reset_backend(monkeypatch):
    """Default: backend_url = None (applies to all tests)"""
    from nanorelay.core import config
    from nanorelay.relay import dispatcher

    monkeypatch.setattr(config.settings, "backend_url", None)
    monkeypatch.setattr(dispatcher, "MODELS_MAP", {
        "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf": {"url": None, "timeout": 60.0}
    })
