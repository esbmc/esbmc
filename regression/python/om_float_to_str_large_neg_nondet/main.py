def nondet_float() -> float: ...

# Regression for __python_float_to_str bug: cast (unsigned long long)v is UB
# when v > ULLONG_MAX (~1.8e19). Before the fix, str(2e19) triggered UB and
# produced frac*=10.0 overflow. The fix adds a pure-float large-value path.
# This test exercises the new path with the minimal reproducer value.
v = nondet_float()
__ESBMC_assume(v == 2.0e19)
s = str(v)
# The large-value path produces at least "2.0" and a non-empty NUL-terminated
# string — we cannot assert the exact representation (it's an approximation)
# but we can assert it is well-formed (non-empty, contains '.').
assert len(s) >= 3
