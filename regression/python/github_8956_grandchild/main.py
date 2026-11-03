class A:

    def func(self):
        return self.child() + 10


class B(A):

    def __init__(self):
        self.child = self.func2

    def func2(self):
        return 1


class C(B):
    pass


assert B().func() == 11
assert C().func() == 11
