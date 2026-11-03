class A:

    def func(self):
        return self.child()


class B(A):

    def __init__(self):
        self.child = self.func2

    def func2(self):
        return 1


a = A()
raised = False
try:
    a.func()
except AttributeError:
    raised = True
assert raised
assert B().func() == 1
