class Box:
    def __init__(self) -> None:
        self.x: int | None = None
        self.flag: int = 7


def main() -> None:
    b = Box()
    _ = b.x
    assert b.flag == 7


main()
