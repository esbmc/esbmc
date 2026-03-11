class A:
    def f(self) -> int:
        return 1

class B:
    def g(self) -> A:
        return A()

x = B().g().f()
assert x == 1
