# Verification harness for the nondet_dict generator
# (src/python-frontend/models/nondet.py, expanded by the preprocessor).
#
# nondet_dict(max_size) returns a dict whose entry count is non-deterministic
# in [0, max_size].  The preprocessor expands the call to an if-chain over
# concrete sequential keys, so the number of live entries is bounded by the
# requested max_size.  This meta-harness verifies that size postcondition.
#
# ENSURES:
#   E1: 0 <= len(nondet_dict(5)) <= 5            [explicit bound honoured]
#   E2: len(nondet_dict(0)) == 0                 [degenerate bound is empty]
d: dict[int, int] = nondet_dict(5)
assert len(d) >= 0  # E1 lower
assert len(d) <= 5  # E1 upper

e: dict[int, int] = nondet_dict(0)
assert len(e) == 0  # E2
