class C:
    def m1(self) -> int:
        return 1


o = C()


def k() -> int:
    return a()


assert k() == 1
a = o.m1
