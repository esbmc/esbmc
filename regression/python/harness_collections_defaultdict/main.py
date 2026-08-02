# Verification harness for collections.defaultdict
# (src/python-frontend/models/collections.py).
#
# defaultdict(int) is modelled as a plain dict whose missing keys are filled
# with the factory default (0 for int), tracked by the preprocessor.
#
# REQUIRES:
#   R1: a nondet value exercises an arbitrary stored entry.
#
# ENSURES:
#   E1: reading an absent key yields the factory default 0
#   E2: an explicitly written key reads back its value
#   E3: a still-absent key keeps returning the default
from collections import defaultdict

x: int = nondet_int()

d: dict[int, int] = defaultdict(int)

assert d[7] == 0        # E1

d[7] = x
assert d[7] == x        # E2

assert d[100] == 0      # E3
