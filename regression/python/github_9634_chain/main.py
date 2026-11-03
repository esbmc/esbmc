class Holder:
    def __init__(self, a):
        self.a = a


class Z:
    def foo(self):
        return 1


class Y(Z):
    def bar(self):
        return 2


class X(Y):
    def baz(self):
        return 3


class W(X):
    def omg(self):
        return 4


h = Holder(W())
assert h.a.omg() == 4
assert h.a.baz() == 3
assert h.a.bar() == 2
assert h.a.foo() == 1
