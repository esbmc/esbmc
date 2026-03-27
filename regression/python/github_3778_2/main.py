class C:

    def h(self) -> int:
        return 42


class A:

    def f(self) -> C:
        return C()


class B:

    def g(self) -> A:
        return A()


x = B().g().f().h()
assert x == 42
