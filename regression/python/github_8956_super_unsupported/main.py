class A:

    def func(self):
        return self.child()


class B(A):

    def __init__(self):
        self.child = self.func2

    def func(self):
        return super().func() + 10

    def func2(self):
        return 1


assert B().func() == 11
