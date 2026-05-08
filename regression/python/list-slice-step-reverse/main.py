# Regression for issue #4293: reverse list slicing with negative step.
xs = [10, 20, 30, 40, 50, 60]

rev = xs[::-1]
assert len(rev) == 6
assert rev[0] == 60
assert rev[1] == 50
assert rev[2] == 40
assert rev[3] == 30
assert rev[4] == 20
assert rev[5] == 10

# Out-of-range positive start with negative step clamps to size-1.
# CPython: xs[100::-1] == xs[::-1].
clamp_high = xs[100::-1]
assert len(clamp_high) == 6
assert clamp_high[0] == 60
assert clamp_high[5] == 10

# Out-of-range negative start with negative step clamps to -1, giving an
# empty slice. CPython: xs[-100::-1] == [].
clamp_low = xs[-100::-1]
assert len(clamp_low) == 0

# Explicit non-negative stop near boundary with negative step.
mid = xs[5:0:-1]
assert len(mid) == 5
assert mid[0] == 60
assert mid[4] == 20

# Negative bound supplied as a runtime variable (not a USub literal). The
# bound resolution must clamp at runtime; otherwise the loop indexes OOB.
# CPython: xs[5:-2:-1] == [xs[5]] == [60].
b = -2
runtime_neg = xs[5:b:-1]
assert len(runtime_neg) == 1
assert runtime_neg[0] == 60
