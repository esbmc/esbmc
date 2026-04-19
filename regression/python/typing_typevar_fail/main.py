from typing import TypeVar

T = TypeVar("T")


def identity(x: T) -> T:
    return x


assert identity(1) == 2
