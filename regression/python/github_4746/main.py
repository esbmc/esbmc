from typing import Optional


class Box:

    def __init__(self, x: Optional[int] = None) -> None:
        self.x = x


def main() -> None:
    b = Box(None)
    if b.x is not None:
        assert b.x == 0  # unreachable: b.x is None

    c = Box(5)
    if c.x is not None:
        assert c.x == 5


main()
