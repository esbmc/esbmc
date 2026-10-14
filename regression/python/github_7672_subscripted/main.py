# A subscripted Callable spells its signature out and keeps it.
from typing import Callable


def apply(g: Callable[[], int]) -> int:
    return g()


def five() -> int:
    return 5


def main() -> None:
    assert apply(five) == 5


main()
