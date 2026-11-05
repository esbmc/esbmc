# Issue #9811: iterating a list literal holding an `X or None` name.
import random

s = 1 or None
for v in [s]:
    assert v == 1

t = "ab" if random.random() < 0.5 else None
for w in [t, t]:
    assert w is None or w == "ab"

xs = [v + 1 for v in [s]]
assert xs[0] == 2
