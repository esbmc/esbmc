def plus(k):
    def dec(f):
        def w(self, x):
            return f(self, x) + k
        return w
    return dec


def double(f):
    def w(self, x):
        return 2 * f(self, x)
    return w


class A:
    @plus(1)
    @double
    @plus(k=10)
    def m(self, x: int) -> int:
        return x


assert A().m(3) == 27
