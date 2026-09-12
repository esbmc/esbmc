# A bare `Callable` parameter resolved to a pointer whose code type returns
# void, so the call through it carried no value and this assertion folded away.
from typing import Callable


def apply(g: Callable):
    return g()


def five() -> int:
    return 5


def main() -> None:
    assert apply(five) == 6


main()
