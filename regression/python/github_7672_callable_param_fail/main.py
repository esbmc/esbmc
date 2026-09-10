# `five` returns 5, so the call through the `Callable`-annotated parameter must
# not prove 6. Guards the return value actually reaching the caller rather than
# staying nondet (#7672).
from typing import Callable


def apply(g: Callable):
    return g()


def five() -> int:
    return 5


def main() -> None:
    assert apply(five) == 6


main()
