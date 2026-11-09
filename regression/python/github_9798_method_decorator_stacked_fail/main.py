def inc(f):
    def w(self) -> int:
        return f(self) + 1
    return w


def dbl(f):
    def w(self) -> int:
        return f(self) * 2
    return w


class C:
    @inc
    @dbl
    def g(self):
        return 2

    @dbl
    @inc
    def h(self):
        return 2


c = C()
assert c.g() == 5
assert c.h() == 5
