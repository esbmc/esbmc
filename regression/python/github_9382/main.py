# Issue #9382: float `%` by zero must raise a catchable ZeroDivisionError.
a: float = nondet_float()
b: float = nondet_float()
__ESBMC_assume(b == 0.0)
ok: bool = False
try:
    r: float = a % b
except ZeroDivisionError:
    ok = True
assert ok

c: float = nondet_float()
__ESBMC_assume(c > 0.5 and c < 4.0)
assert 5.5 % 2.0 == 1.5
assert 7.0 % c >= 0.0
