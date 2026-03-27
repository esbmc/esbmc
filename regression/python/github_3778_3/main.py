class A:

    def f(self, n: int) -> int:
        return n + 1


class B:

    def g(self) -> A:
        return A()


x = B().g().f(5)
assert x == 6
