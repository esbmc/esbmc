# A parameter annotated with an unsubscripted `Callable` must not lose the
# value of the call made through it. An unsubscripted `Callable` is
# `Callable[..., Any]` (PEP 484); typing its return void made `apply` look
# like it returned None and folded the comparison to False (#7672).
from typing import Callable


def apply(g: Callable):
    return g()


def five() -> int:
    return 5


def main() -> None:
    assert apply(five) == 5


main()
