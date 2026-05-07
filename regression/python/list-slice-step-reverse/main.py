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
