class A:
    def f(self, x: int) -> int:
        return x


class B(A):
    def f(self, x: int, y: int) -> int:
        return x


class H:
    def __init__(self) -> None:
        self.p: A = B()


h = H()
assert h.p.f(1, 2) == 1
