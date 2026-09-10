# The `-> int` counterpart of github_7672_callable_param_fail: `five` returns
# 5, so 6 must not be provable (#7672).
from typing import Callable


def apply(g: Callable) -> int:
    return g()


def five() -> int:
    return 5


def main() -> None:
    assert apply(five) == 6


main()
