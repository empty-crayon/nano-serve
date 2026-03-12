import pytest
from nanorelay.relay.dispatcher import Dispatcher
from nanorelay.core.exceptions import InvalidRequestError

def test_unsupported_model_raises(reset_backend):
    dispatcher = Dispatcher()
    with pytest.raises(InvalidRequestError):
        dispatcher._resolve_backend("not-registered-model")