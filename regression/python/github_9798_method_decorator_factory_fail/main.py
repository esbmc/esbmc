def scale(k, offset=4):
    def dec(f):
        def w(self, x):
            return k * f(self, x) + offset
        return w
    return dec


class A:
    @scale(3)
    def m(self, x: int) -> int:
        return x

    @scale(k=2, offset=0)
    def n(self, x: int) -> int:
        return x


class B:
    @scale(5)
    def m(self, x: int) -> int:
        return x


x = nondet_int()
__ESBMC_assume(-1000 < x < 1000)
assert A().m(x) == 3 * x + 4
assert A().n(x) == 2 * x
assert B().m(x) == 3 * x + 4
