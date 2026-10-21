class A:

    def __init__(self) -> None:
        self.items: list[float] = []
        self.n: list[int] = [7]

    def total(self) -> float:
        s = 0.0
        for v in self.items:
            s += v
        return s

    def copy(self) -> "A":
        r = A()
        for v in self.items:
            r.items.append(v)
        return r


def main() -> None:
    a = A()
    a.items.append(1.5)
    a.items.append(2.25)
    for v in a.items:
        assert v > 1.0
    assert a.total() == 3.75
    b = a.copy()
    assert b.items[1] == 2.25
    assert b.total() == 3.75


main()
