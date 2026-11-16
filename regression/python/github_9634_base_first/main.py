class Z:
    def foo(self):
        return 1


class Y(Z):
    def bar(self):
        return 2


class C:
    def __init__(self, a):
        self.a = a


c = C(Y())
assert c.a.foo() == 1
