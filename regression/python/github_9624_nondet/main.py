# Issue #9624
class Point:
    def __init__(self, i):
        self.x = i * 1.5

    def maximize(self, other):
        if self.x < other.x:
            self.x = other.x
        return self


def maximize(points):
    nxt = points[0]
    for p in points[1:]:
        nxt = nxt.maximize(p)
    return nxt


a = nondet_int()
b = nondet_int()
__ESBMC_assume(0 <= a and a <= 10 and 0 <= b and b <= 10)
r = maximize([Point(a), Point(b), Point(4)])
assert r.x >= a * 1.5 and r.x >= b * 1.5 and r.x >= 6.0
