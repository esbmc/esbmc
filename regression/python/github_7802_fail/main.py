# A bare `Callable` return carries the signature of the functions the body
# returns, both for a call through a bound variable and for `pick(c)(x)` (#7802).
from typing import Callable


def inc(m: int) -> int:
    return m + 1


def century(m: int) -> int:
    return m + 100


def pick(c: bool) -> Callable:
    return inc if c else century


def main() -> None:
    assert pick(True)(1) == 3
    h = pick(False)
    assert h(1) == 101


main()
