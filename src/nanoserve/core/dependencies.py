from fastapi import Request
from nanoserve.relay.dispatcher import Dispatcher


def get_dispatcher(request: Request) -> Dispatcher:
    return request.app.state.dispatcher