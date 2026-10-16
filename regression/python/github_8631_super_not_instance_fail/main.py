class A:

    def f(self):
        return 1


class B(A):

    def f(self):
        return 2


class C(B):
    pass


class D(B):

    def f(self):
        return super(C, self).f()


D().f()
