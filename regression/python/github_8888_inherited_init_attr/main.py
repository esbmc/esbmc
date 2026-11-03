class A:
    def __init__(self, v):
        self.a = v


class B(A):
    pass


class C(B):
    pass


class M:
    pass


class D(A, M):
    pass


b = B(1)
assert isinstance(b.a, int)
assert type(b.a) is int
c = C(2)
assert isinstance(c.a, int)
d = D(3)
assert isinstance(d.a, int)
assert d.a == 3
