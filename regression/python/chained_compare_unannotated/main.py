# A chained comparison (a <= x <= b) whose middle operand is an unannotated
# parameter crashed ESBMC: the parameter is typed as a pointer, and while the
# first comparison was type-reconciled, the loop that builds the remaining
# comparisons left a pointer-vs-int compare that aborted the SMT backend
# (convert_ptr_cmp). The pointer operand is now reconciled like the first.
class Range:
    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi

    def check(self, x):
        return self.lo <= x <= self.hi


r = Range(1, 10)
assert r.check(5)
assert r.check(1)
assert r.check(10)
assert not r.check(0)
assert not r.check(20)


# Unannotated free-function parameters in a chained comparison.
def between(a, b, c):
    return a <= b <= c


assert between(1, 2, 3)
assert not between(3, 2, 1)


# One bound a literal, the other a self attribute.
class Upper:
    def __init__(self, hi):
        self.hi = hi

    def ok(self, x):
        return 0 <= x <= self.hi


u = Upper(10)
assert u.ok(5)
assert not u.ok(11)
