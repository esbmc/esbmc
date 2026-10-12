class C:
    def m1(self) -> int:
        return 1


o = C()


def h() -> None:
    a, n = o.m1, 0


def k() -> int:
    return a()


h()
assert k() == 1
