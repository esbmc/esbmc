class A:
    def func(self):
        return "a"


class B(A):
    pass


class C(B):
    pass


class D(B):
    def func(self):
        return len(super(C, self).func())


D().func()
