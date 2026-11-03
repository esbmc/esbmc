class A:
    def __init__(self):
        self.v = 1


class B(A):
    pass


class C(A):
    def __init__(self):
        self.v = 3


class D(B, C):
    pass


d = D()
assert d.v == 1
