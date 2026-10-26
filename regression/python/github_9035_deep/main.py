class A:
    def func(self, k: int) -> int:
        return self.g() + k

    def g(self) -> int:
        return 0


class B(A):
    pass


class C(B):
    def g(self) -> int:
        return 10


class D(C):
    pass


assert A().func(1) == 1
assert B().func(1) == 1
assert C().func(1) == 11
assert D().func(1) == 11
