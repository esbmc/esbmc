class C:

    def __init__(self, v: int) -> None:
        self.v = v


xs = [C(1), C(2)]
n = xs.count(C(1))
assert n == 1
