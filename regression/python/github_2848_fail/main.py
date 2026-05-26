from typing import Any


def f(c: int) -> Any:
    if c > 0:
        return 1
    return "negative"


def main() -> None:
    x = f(5)
    # f(5) returns 1, so this assertion is false.
    assert x == 2


main()
