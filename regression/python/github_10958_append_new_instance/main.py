class C:

    def __init__(self, v: int) -> None:
        self.v = v


xs = [C(1), C(2)]
r = xs.append(C(3))
assert r is None
assert len(xs) == 3
