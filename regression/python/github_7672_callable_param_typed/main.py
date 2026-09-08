# Same shape with a return annotation on the caller: the void return of the
# unsubscripted `Callable` reached the solver as `Unexpected type in int/ptr
# typecast` instead of a verdict (#7672).
from typing import Callable


def apply(g: Callable) -> int:
    return g()


def five() -> int:
    return 5


def main() -> None:
    assert apply(five) == 5


main()
