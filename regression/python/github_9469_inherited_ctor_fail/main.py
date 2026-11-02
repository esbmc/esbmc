class Z:
    def foo(self):
        return 1


class Y(Z):
    pass


class C:
    def __init__(self, a):
        self.a = a


class D(C):
    pass


d = D(Y())
assert d.a.foo() == 3
