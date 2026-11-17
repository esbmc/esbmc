# Issue #10490: a repeat count below one gives an empty list, however large.
n = -4611686018427387899
x = [5] * n
assert len(x) == 5
