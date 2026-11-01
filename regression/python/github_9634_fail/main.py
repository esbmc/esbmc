class C:
    def __init__(self, a):
        self.a = a


class Z:
    def foo(self):
        return 1


class Y(Z):
    def bar(self):
        return 2


c = C(Y())
assert c.a.foo() == 2
