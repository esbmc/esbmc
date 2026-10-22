class C:

    def __init__(self, v: int) -> None:
        self.v = v

    def get(self) -> int:
        return self.v


class A:

    def __init__(self, u: bool) -> None:
        self.c = C(1) if u else C(2)

    def f(self) -> int:
        return self.c.get()


u = nondet_bool()
assert A(u).f() == (1 if u else 2)
