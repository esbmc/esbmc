# The default y=2 makes f(3) == 6, not 99.
f = lambda x, y=2: x * y
assert f(3) == 99
