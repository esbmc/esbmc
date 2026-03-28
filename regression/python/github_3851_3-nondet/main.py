# Two nondet int globals declared after a class with two methods.
# Verifies both globals are visible inside the methods under all nondet values.

class Range:
    def in_range(self, x: int) -> bool:
        return lo <= x <= hi

    def width(self) -> int:
        return hi - lo


lo: int = nondet_int()
hi: int = nondet_int()
__ESBMC_assume(lo >= 0)
__ESBMC_assume(hi >= lo)

r = Range()
assert r.width() >= 0
assert r.in_range(lo)
assert r.in_range(hi)
