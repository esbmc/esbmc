# Regression for issue #4293: list slicing with step.
# Pre-fix: xs[::2] silently ignored the step value and returned the full list.
xs = [10, 20, 30, 40, 50, 60]

ys = xs[::2]
assert len(ys) == 3
assert ys[0] == 10
assert ys[1] == 30
assert ys[2] == 50

zs = xs[1::2]
assert len(zs) == 3
assert zs[0] == 20
assert zs[1] == 40
assert zs[2] == 60
