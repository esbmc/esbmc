class A:
    def func(self):
        return self.g()

    def g(self):
        return 0


class B(A):
    def func(self):
        return 100

    def g(self):
        return 1


class C(A):
    def g(self):
        return 2


assert B().func() == 100
assert C().func() == 2
assert A().func() == 0
