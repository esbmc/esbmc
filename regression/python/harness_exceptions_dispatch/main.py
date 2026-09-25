# Verification harness for exception dispatch
# (src/python-frontend/models/exceptions.py; symbolic exception lowering,
# docs/design-symbolic-exceptions.md).
#
# A nondet integer decides which exception type is raised, and the harness
# checks that the matching handler runs and correlates with the path taken.
#
# REQUIRES:
#   R1: a non-deterministic integer selects the raise site, exploring both
#       the ValueError and TypeError paths.
#
# ENSURES (which records the handler that ran):
#   E1: exactly one handler runs (which is 1 or 2)   [some handler catches]
#   E2: the ValueError handler runs iff x > 0         [dispatch matches the
#       raised type, which is tied to the path condition]
x: int = nondet_int()
which: int = 0

try:
    if x > 0:
        raise ValueError("pos")
    else:
        raise TypeError("nonpos")
except ValueError:
    which = 1
    assert x > 0
except TypeError:
    which = 2
    assert x <= 0

assert which == 1 or which == 2  # E1
assert (which == 1) == (x > 0)  # E2
