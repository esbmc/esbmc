# Verification harness for the nondet_list generator
# (src/python-frontend/models/nondet.py, expanded by the preprocessor).
#
# nondet_list(max_size) returns a list whose length is non-deterministic in
# the closed range [0, max_size] (see _nondet_size, which assumes size >= 0
# and size <= max_size).  This is a *meta*-harness: it verifies the model's
# own documented size postcondition rather than user code.
#
# REQUIRES:
#   (none) — the generator takes a concrete max_size; the length is the only
#   non-deterministic quantity.
#
# ENSURES:
#   E1: 0 <= len(nondet_list(5)) <= 5            [explicit bound honoured]
#   E2: 0 <= len(nondet_list()) <= 8             [default max_size == 8]
#   E3: len(nondet_list(0)) == 0                 [degenerate bound is empty]
xs: list[int] = nondet_list(5)
assert len(xs) >= 0  # E1 lower
assert len(xs) <= 5  # E1 upper

ys: list[int] = nondet_list()
assert len(ys) >= 0  # E2 lower
assert len(ys) <= 8  # E2 upper

zs: list[int] = nondet_list(0)
assert len(zs) == 0  # E3
