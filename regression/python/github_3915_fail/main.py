# Failing variant of issue #3915: same value-type inference path, but the
# top-level assertion is wrong.
from collections import defaultdict


def f():
    d = defaultdict(int)
    d[0, 0] = 5
    return min(d[0, 0], 7)


assert f() == 99
