def nondet_float() -> float: ...

# __python_float_to_str: verify output format for a nondet float
# constrained to a single-point value at a time.
# Three subproblems for 0.0, 1.0, and 3.5 — the three stated constants.

v = nondet_float()
__ESBMC_assume(v == 0.0)
s0 = str(v)
assert s0 == "0.0"

v2 = nondet_float()
__ESBMC_assume(v2 == 1.0)
s1 = str(v2)
assert s1 == "1.0"

v3 = nondet_float()
__ESBMC_assume(v3 == 3.5)
s3 = str(v3)
assert s3 == "3.5"
