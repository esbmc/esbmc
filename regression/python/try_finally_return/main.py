# `return` inside a try/finally runs the finally on its way out, and the
# returned expression is evaluated before the finally does. Issue #7076: this
# used to be refused during conversion.


class Box:
    def __init__(self) -> None:
        self.v: int = 0


def cleanup_runs(b: Box) -> int:
    try:
        return 1
    finally:
        b.v = 7


def value_is_evaluated_first() -> int:
    x: int = 1
    try:
        return x
    finally:
        x = 2


def main() -> None:
    b = Box()
    assert cleanup_runs(b) == 1
    assert b.v == 7
    assert value_is_evaluated_first() == 1


main()
