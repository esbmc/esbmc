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


h = Holder(X())
assert h.a.baz() == 1
