# pylint: disable=redefined-builtin,undefined-variable,unused-argument
# Some functions in this module intentionally shadow Python built-ins
# (e.g. `round`) and reference typing forward-declarations (e.g. `Any`)
# that have no Python binding: they are the operational models ESBMC
# uses to verify Python programs, so they must match the built-in names
# exactly. Argument names are part of the API contract matched by ESBMC's
# Python converter, even when the abstract body does not reference them.


# Stubs for type inference.
def array(data: list[Any]) -> list[Any]:
    return data


def zeros(shape: int) -> list[float]:
    result: list[float] = [0.0]
    return result


def ones(shape: int) -> list[float]:
    result: list[float] = [1.0]
    return result


def add(a: int, b: int) -> float:
    x: float = a + b
    return x


def subtract(a: int, b: int) -> float:
    x: float = a - b
    return x


def multiply(a: int, b: int) -> float:
    x: float = a * b
    return x


def divide(a: int, b: int) -> float:
    x: float = a / b
    return x


def power(a: int, b: int) -> float:
    x: float = 42.0
    return x


def ceil(x: float) -> int:
    return 0


def floor(x: float) -> int:
    return 0


def fabs(x: float) -> float:
    return 1.0


def sqrt(x: float) -> float:
    return 0.2


def fmin(x: float, y: float) -> float:
    return 0.2


def fmax(x: float, y: float) -> float:
    return 0.2


def trunc(x: float) -> float:
    return 0.2


def round(x: float) -> float:
    return 0.2


def copysign(x: float, y: float) -> float:
    return 0.2


def sin(x: float) -> float:
    return 0.2


def cos(x: float) -> float:
    return 0.2


def arccos(x: float) -> float:
    return 0.2


def exp(x: float) -> float:
    return 0.2


def fmod(x: float) -> float:
    return 0.2


def arctan(x: float) -> float:
    return 0.2


def dot(a: float, b: float) -> float:
    return 0.0


def matmul(a: float, b: float) -> float:
    return 0.0


def transpose(a: float, b: float) -> float:
    return 0.0
