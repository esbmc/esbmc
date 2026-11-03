class P:
    def __init__(self, x: float):
        self.x = x


class Q:
    def __init__(self, y: int):
        self.y = y


def larger(a, b):
    m = a.x if a.x > b.y else b.y
    return m


n = nondet_int()
__ESBMC_assume(n > 0 and n < 10)
r = larger(P(2.5), Q(n))
assert r >= 2.5
assert r == 2.5 or r == n


def larger_same(a, b):
    m = a.y if a.y > b.y else b.y
    return m


assert larger_same(Q(1), Q(2)) == 2
