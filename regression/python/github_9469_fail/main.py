class Z:
    def foo(self):
        return 1


class Y(Z):
    pass


class C:
    def __init__(self, a):
        self.a = a


c = C(Y())
assert c.a.foo() == 2
