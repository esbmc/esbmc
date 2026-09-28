# `for i, v in enumerate(pairs)` was lowered to a while loop whose value
# variable is a bare-`tuple`-annotated subscript read, carrying no component
# types, so a later `a, b = v` was handed an untyped value and refused to
# unpack. Unrolling binds each tuple literal directly and keeps its shape, as
# the nested-target form already did.
from typing import List, Tuple

pairs = [(1, 2), (3, 4)]
total = 0
for i, tpl in enumerate(pairs):
    a, b = tpl
    assert a < b
    total = total + a + b
assert total == 10
assert i == 1

# An explicit annotation takes the same path.
typed: List[Tuple[int, int]] = [(5, 6), (7, 8)]
for j, t2 in enumerate(typed):
    c, d = t2
    assert d == c + 1

# A non-zero start is honoured.
for k, t3 in enumerate(pairs, 2):
    e, f = t3
    assert e < f
assert k == 3

# Lists of non-tuple elements keep the ordinary lowering.
nums = [10, 20, 30]
seen = 0
for m, n in enumerate(nums):
    seen = seen + n
assert seen == 60
assert m == 2
