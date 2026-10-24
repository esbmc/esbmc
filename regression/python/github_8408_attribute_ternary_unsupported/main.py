class C:

    def f(self):
        return "c"


class D:

    def f(self):
        return 1


class A:

    def __init__(self, u):
        self.c = C() if u else D()

    def f(self):
        return self.c.f()


r = A(nondet_bool()).f()
