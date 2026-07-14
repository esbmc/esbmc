# GitHub #5105: a user function returning a float comparison (a bool) yielded a
# wrong verdict (FAILED) instead of SUCCESSFUL. Same root cause as #5104 — the
# helper's parameters defaulted to a pointer type because b()'s return type
# could not be inferred from its local `return v`. Since a and c are the same
# expression, math.fabs(a - c) == 0.0 <= 1e-08 + 1e-05 * math.fabs(c).
import math


def allclose1(a, c):
    return math.fabs(a - c) <= 1e-08 + 1e-05 * math.fabs(c)


def b():
    v = nondet_float()
    __ESBMC_assume(v >= -10.0)
    __ESBMC_assume(v <= 10.0)
    return v


p = b()
q = b()
a = p * q
c = p * q
assert allclose1(a, c)
