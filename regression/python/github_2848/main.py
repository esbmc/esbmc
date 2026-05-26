from typing import Any


def f(c: int) -> Any:
    # Mixed-type Any return: a string sibling return must not abort
    # conversion (issue #2848); the int return drives inference.
    if c > 0:
        return 1
    return "negative"


def main() -> None:
    x = f(5)
    assert x == 1


main()
