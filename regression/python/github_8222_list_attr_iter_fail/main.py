class A:

    def __init__(self) -> None:
        self.items: list[float] = []
        self.n: list[int] = [7]

    def total(self) -> float:
        s = 0.0
        for v in self.items:
            s += v
        return s


def main() -> None:
    a = A()
    a.items.append(1.5)
    a.items.append(2.25)
    assert a.total() == 3.75
    for v in a.items:
        assert v != 2.25


main()
