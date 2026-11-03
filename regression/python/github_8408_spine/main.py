class A:

    def f(self):
        return 1


class B:

    def f(self):
        return 2


a = A()
a = B()
assert a.f() == 2
