from fastapi import Request
from nanorelay.relay.dispatcher import Dispatcher

def get_dispatcher(request: Request) -> Dispatcher:
    return request.app.state.dispatcher
