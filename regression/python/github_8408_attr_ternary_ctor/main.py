class C:
    def __init__(self, v: int) -> None:
        self.v = v

    def get(self) -> int:
        return self.v


class A:
    def __init__(self, u: bool) -> None:
        self.c = C(1) if u else C(2)

    def get(self) -> int:
        return self.c.get()


u = nondet_bool()
a = A(u)
assert a.get() == (1 if u else 2)
assert a.c.v == a.get()
