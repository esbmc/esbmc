class C:
    def __init__(self, base: int) -> None:
        self.base = base

    def add(self, x: int) -> int:
        return self.base + x


o = C(10)


class D:
    def run(self, y: int) -> int:
        return f(y)


def k(y: int) -> int:
    return f(y) * 2


f = o.add
o.base = 20
assert k(1) == 42
assert D().run(2) == 22
