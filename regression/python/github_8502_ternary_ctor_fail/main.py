class B:
    def __init__(self, v: int):
        self.a = v

    def get(self) -> int:
        return self.a


def pick(c: bool) -> int:
    b = B(1) if c else B(2)
    return b.get()


c = nondet_bool()
b = B(v=1) if c else B(2)
assert b.get() == (1 if c else 2)
n = None if c else B(3)
if c:
    assert n is None
else:
    assert n.a == 4
assert pick(True) == 1
assert pick(False) == 2
