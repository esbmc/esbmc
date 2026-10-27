class B:

    def __init__(self, v: int):
        self.a = v

    def get(self) -> int:
        return self.a


def pick(c: bool) -> int:
    if c:
        b = B(1)
    else:
        b = B(2)
    return b.get()


c = nondet_bool()
n = None if c else B(3)
if not c:
    if n is None:
        raise ValueError("x")
assert pick(True) == 2
