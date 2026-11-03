# GitHub #9015: isinstance/hasattr on a call's result evaluate the call once.
class P:
    def __init__(self):
        self.x = 1


class R:
    def __init__(self):
        self.n = 0

    def m(self):
        return 1

    def count(self) -> int:
        self.n += 1
        return self.n

    def p(self) -> P:
        return P()


assert isinstance(R().m(), int)
assert not isinstance(R().m(), str)
r = R()
assert isinstance(r.count(), (str, float, int))
assert r.n == 1
assert hasattr(r.p(), "x")
