class BaseException:
    message: str = ""

    def __init__(self, message: str):
        self.message = message

    def __str__(self) -> str:
        return self.message


class ValueError(BaseException):
    message: str = ""

    def __init__(self, message: str):
        self.message = message

    def __str__(self) -> str:
        return self.message


class TypeError(BaseException):
    message: str = ""

    def __init__(self, message: str):
        self.message = message

    def __str__(self) -> str:
        return self.message


class IndexError(BaseException):
    message: str = ""

    def __init__(self, message: str):
        self.message = message

    def __str__(self) -> str:
        return self.message


class KeyError(BaseException):
    message: str = ""

    def __init__(self, message: str):
        self.message = message

    def __str__(self) -> str:
        return self.message

class ZeroDivisionError(BaseException):
    message: str = ""

    def __init__(self, message: str):
        self.message = message

    def __str__(self) -> str:
        return self.message
