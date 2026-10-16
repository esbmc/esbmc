class A:
    def __init__(self):
        self.v = 1

    def func(self) -> int:
        return 1


class B(A):
    pass


class C(A):
    def __init__(self):
        self.v = 3

    def func(self) -> int:
        return 3


class D(B, C):
    pass


d = D()
assert d.func() == 3
assert d.v == 3
b = B()
assert b.func() == 1
assert b.v == 1
