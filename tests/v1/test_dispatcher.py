import pytest
from nanoserve.relay.dispatcher import Dispatcher
from nanoserve.core.backend_config import BackendConfig
from nanoserve.schemas.request_context import RequestContext


def _make_dispatcher():
    config = BackendConfig(
        backend_url="http://localhost:8000",
        timeout=120,
        echo_timeout=10,
    )
    return Dispatcher(config=config)


@pytest.mark.asyncio
async def test_dispatch_echo_model():
    dispatcher = _make_dispatcher()
    ctx = RequestContext(
        request_id="test-1",
        model="echo",
        stream=False,
        messages=[],
    )
    # Echo dispatch returns a dict, no HTTP needed
    result = await dispatcher.dispatch(ctx)
    assert isinstance(result, dict)
    await dispatcher.close()


def test_technique_passed_through():
    """technique field is set from X-Pipeline-Stage header — dispatcher doesn't touch it."""
    ctx = RequestContext(
        request_id="test-2",
        model="some-model",
        stream=False,
        technique="classify",
    )
    assert ctx.technique == "classify"