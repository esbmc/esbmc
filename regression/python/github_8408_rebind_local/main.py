class A:

    def __init__(self) -> None:
        self.x = 1

    def f(self) -> int:
        return self.x


class C:

    def __init__(self) -> None:
        self.y = 2

    def f(self) -> int:
        return self.y + 10


def main() -> None:
    a = A()
    r1 = a.f()
    a = C()
    a.y = 5
    r2 = a.f()
    assert not isinstance(a, A)
    a = A()
    assert r1 == 1
    assert r2 == 15
    assert a.f() == 1


main()
