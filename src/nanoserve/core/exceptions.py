class NanoServeException(Exception):
    """Base exception"""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message


class InvalidRequestError(NanoServeException):
    def __init__(self, message: str):
        super().__init__(status_code=400, message=message)