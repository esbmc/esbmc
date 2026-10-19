class A:
    def __init__(self, v):
        self.a = v


class B(A):
    pass


class C(B):
    pass


c = C(2)
assert isinstance(c.a, int)
assert isinstance(c.a, str)
