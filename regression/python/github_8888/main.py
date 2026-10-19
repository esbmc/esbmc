class A:
    def __init__(self, v):
        self.a = v


class B(A):
    pass


class C(B):
    pass


class S:
    def __init__(self, s):
        self.s = s


class T(S):
    pass


b = B(1)
assert isinstance(b.a, int)
assert type(b.a) is int
assert b.a + 1 == 2

c = C(2)
assert isinstance(c.a, int)
assert c.a == 2

t = T("xy")
assert isinstance(t.s, str)
assert t.s == "xy"
