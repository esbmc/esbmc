# Chained assignment with tuple unpacking should fail assertion.
# After (x, y) = (u, v) = f(), x and u should both be 1, not 2.


def f():
    return (1, 2)


(x, y) = (u, v) = f()

assert x == 2  # should fail: x is 1
