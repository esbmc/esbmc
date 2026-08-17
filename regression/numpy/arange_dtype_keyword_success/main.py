import numpy as np

# A dtype= keyword is not modeled by the literal-materialization fast path;
# it must fall back to the operational model instead of being silently
# ignored or rejected outright.
a = np.arange(3, dtype=np.int32)
assert len(a) == 3
assert a[2] == 2
