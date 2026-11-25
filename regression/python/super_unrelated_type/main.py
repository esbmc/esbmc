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


raised = False
try:
    D().func()
except TypeError:
    raised = True
assert raised
