class A:

    def __init__(self, v: int) -> None:
        self.v = v

    def m(self) -> int:
        return self.v


class B(A):
    pass


c = nondet_bool()
cls = B if c else A
o = cls(3)
assert o.m() == 3
assert isinstance(o, A)
assert isinstance(o, B) == c
