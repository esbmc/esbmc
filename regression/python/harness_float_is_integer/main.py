# Verification harness for float.is_integer (src/python-frontend/models/float.py).
#
# x.is_integer() returns True iff the float x has no fractional part; the model
# computes this as x == int(x) (int truncates toward zero).
#
# REQUIRES:
#   R1: a nondet float x, bounded to a finite interval so int(x) is well-defined.
#   R2: a nondet int n, bounded, to build an integral float.
#
# ENSURES:
#   E1: is_integer() agrees with the truncation test: r == (x == int(x))
#   E2: is_integer() True implies the value equals its truncation
#   E3: a float built from an int is integral
#
# Note: float(n).is_integer() is split across a local (y) because the frontend
# mis-lowers the chained call float(n).is_integer() into a TypeError.
x: float = nondet_float()
__ESBMC_assume(-1000.0 <= x <= 1000.0)

r: bool = x.is_integer()
assert r == (x == int(x))  # E1
if r:
    assert x == int(x)  # E2

n: int = nondet_int()
__ESBMC_assume(-1000 <= n <= 1000)
y: float = float(n)
assert y.is_integer()  # E3

# Concrete anchors.
assert (5.0).is_integer()
assert not (5.5).is_integer()
assert (0.0).is_integer()
