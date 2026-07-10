# Falsification harness for the nondet_list generator
# (src/python-frontend/models/nondet.py).
#
# Validates that the size lower bound is really 0 (the list may be empty):
# a harness that assumes a non-empty result must be falsified.
#
# WRONG PROPERTY (expected to be falsified):
#   F1: len(nondet_list(5)) >= 1.  False because _nondet_size admits size 0,
#       so the empty list is a legal outcome.
xs: list[int] = nondet_list(5)
assert len(xs) >= 1          # F1 — falsifiable (empty list is allowed)
