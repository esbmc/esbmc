class A:

    def __init__(self, v: int) -> None:
        self.v = v

    def f(self) -> int:
        return self.v


class B:
    pass


def pick(c: bool) -> int:
    a = A(1) if c else A(2)
    return a.f() + a.v


assert pick(False) == 2
