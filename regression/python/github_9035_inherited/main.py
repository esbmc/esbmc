class A:
    def func(self):
        return self.g()

    def g(self):
        return 0


class B(A):
    def g(self):
        return 1


assert B().func() == 1
x = B().func()
assert x == 1
assert A().func() == 0
assert B().g() == 1
