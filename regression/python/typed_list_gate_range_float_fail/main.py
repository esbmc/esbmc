# range() of a float raises TypeError, so the list keeps the untyped model.
x = 2.0
xs = list(range(x))
assert xs == [0, 1]
