class A:

    def __init__(self) -> None:
        self.items: list[float] = []
        self.n: list[int] = [7]
        self.items.append(1.5)


def main() -> None:
    a = A()
    assert a.items[0] != 1.5


main()
