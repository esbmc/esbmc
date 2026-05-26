from typing import Optional


class Box:

    def __init__(self, x: Optional[int] = None) -> None:
        self.x = x


def main() -> None:
    b = Box(5)
    if b.x is not None:
        assert b.x == 7  # must fail: actual value is 5


main()
