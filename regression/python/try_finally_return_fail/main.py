# The finally really does run on the return path: asserting that it did not is
# detected.


class Box:
    def __init__(self) -> None:
        self.v: int = 0


def f(b: Box) -> int:
    try:
        return 1
    finally:
        b.v = 7


def main() -> None:
    b = Box()
    assert f(b) == 1
    assert b.v == 0


main()
