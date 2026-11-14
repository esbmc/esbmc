class C:
    def m1(self) -> int:
        return 1


o = C()


def k() -> int:
    return a()


a = o.m1
assert k() == 2
