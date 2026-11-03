class A:

    def func(self):
        return self.child()


class B(A):

    def __init__(self):
        self.child = self.func2

    def func2(self):
        return 1


b = B()
assert b.func() == 1
