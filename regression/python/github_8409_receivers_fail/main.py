# A @staticmethod binds no receiver whichever instance expression calls it, and
# an ordinary method in the same class still binds self (#8409).
class C:
    def __init__(self, v: int) -> None:
        self.v = v

    @staticmethod
    def add(x: int, y: int) -> int:
        return x + y

    def via_self(self, y: int) -> int:
        return self.add(self.v, y)

    def plus(self, y: int) -> int:
        return self.v + y


class Holder:
    def __init__(self) -> None:
        self.c = C(100)


c = C(5)
h = Holder()
assert C(0).add(2, 3) == 5
assert c.add(2, 3) == 5
assert c.via_self(1) == 6
assert C(7).via_self(1) == 8
assert h.c.add(4, 5) == 10
assert h.c.via_self(1) == 101
assert C(3).plus(1) == 4
assert c.plus(2) == 7
