class Base:

    def __init__(self) -> None:
        self.items: list[float] = []


class A(Base):

    def __init__(self) -> None:
        super().__init__()
        self.n: list[int] = [7]
        self.items.append(1.5)
        self.items.append(2.5)

    def get(self, i: int) -> float:
        return self.items[i]


def main() -> None:
    a = A()
    assert a.items[0] == 1.5
    assert a.items[-1] == 2.5
    assert a.get(1) == 2.5
    x: float = a.items[0]
    assert x == 1.5
    assert a.n[0] == 7


main()
