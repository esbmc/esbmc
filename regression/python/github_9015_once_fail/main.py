# GitHub #9015: a tuple of types still evaluates the call once.
class R:
    def __init__(self):
        self.n = 0

    def count(self) -> int:
        self.n += 1
        return self.n


r = R()
assert isinstance(r.count(), (str, float, int))
assert r.n == 2
