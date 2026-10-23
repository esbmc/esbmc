class A:
    def func(self):
        return self.g()

    def g(self):
        return 5


class B(A):
    def g(self):
        return 7


assert B().func() == 5
