class A:
    def __init__(self):
        self.v = 1


class B(A):
    def __init__(self):
        self.v = 1
        self.b = 10


class C(A):
    def __init__(self):
        self.v = 3
        self.c = 30


class D(B, C):
    pass


d = D()
assert d.v == 1
assert d.c == 10
