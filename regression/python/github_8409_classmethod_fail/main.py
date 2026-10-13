# A @classmethod called through an instance binds the class, not the instance,
# whichever instance expression calls it (#8409).
class C:
    k: int = 3

    @classmethod
    def bump(cls, x: int) -> int:
        return x + cls.k

    def via_self(self, y: int) -> int:
        return self.bump(y)


class D(C):
    pass


class Holder:
    def __init__(self) -> None:
        self.c = C()


c = C()
h = Holder()
assert C().bump(1) == 4
assert c.bump(2) == 5
assert c.via_self(3) == 6
assert h.c.bump(4) == 7
assert D().bump(5) == 9
