class BaseException:
    message: str = ""

    def __init__(self, message: str):
        self.message: str = message

    def __str__(self) -> str:
        return self.message


class ValueError(BaseException):
    message: str = ""

    def __init__(self, message: str):
        self.message: str = message

    def __str__(self) -> str:
        return self.message


class TypeError(BaseException):
    message: str = ""

    def __init__(self, message: str):
        self.message: str = message

    def __str__(self) -> str:
        return self.message


class IndexError(BaseException):
    message: str = ""

    def __init__(self, message: str):
        self.message: str = message

    def __str__(self) -> str:
        return self.message


class KeyError(BaseException):
    message: str = ""

    def __init__(self, message: str):
        self.message: str = message

    def __str__(self) -> str:
        return self.message


class ZeroDivisionError(BaseException):
    message: str = ""

    def __init__(self, message: str):
        self.message: str = message

    def __str__(self) -> str:
        return self.message


class AssertionError(BaseException):
    message: str = ""

    def __init__(self, message: str):
        self.message: str = message

    def __str__(self) -> str:
        return self.message


class Exception(BaseException):
    message: str = ""

    def __init__(self, message: str):
        self.message: str = message

    def __str__(self) -> str:
        return self.message


class NameError(BaseException):
    message: str = ""

    def __init__(self, message: str):
        self.message: str = message

    def __str__(self) -> str:
        return self.message


class OSError(Exception):
    """Base class for I/O related errors"""
    message: str = ""

    def __init__(self, message: str = "OS error"):
        self.message: str = message

    def __str__(self) -> str:
        return self.message


class FileNotFoundError(OSError):
    """Raised when a file or directory is not found"""
    message: str = ""

    def __init__(self, message: str = "File not found"):
        self.message: str = message

    def __str__(self) -> str:
        return self.message


class FileExistsError(OSError):
    """Raised when trying to create a file or directory that already exists"""
    message: str = ""

    def __init__(self, message: str = "File exists"):
        self.message: str = message

    def __str__(self) -> str:
        return self.message


class PermissionError(OSError):
    """Raised when an operation lacks the necessary permissions"""
    message: str = ""

    def __init__(self, message: str = "Permission denied"):
        self.message: str = message

    def __str__(self) -> str:
        return self.message


class RuntimeError(Exception):
    """Base class for runtime errors"""
    message: str = ""

    def __init__(self, message: str = "Runtime error"):
        self.message: str = message

    def __str__(self) -> str:
        return self.message


class NotImplementedError(RuntimeError):
    """Raised when an abstract method requires implementation"""
    message: str = ""

    def __init__(self, message: str = "Method not implemented"):
        self.message: str = message

    def __str__(self) -> str:
        return self.message
