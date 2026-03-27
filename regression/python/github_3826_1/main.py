# Chained assignment with tuple unpacking: (x, y) = (u, v) = f()
# Both tuple targets get unpacked from the same RHS evaluated once.


def f():
    return (1, 2)


(x, y) = (u, v) = f()

assert x == 1
assert y == 2
assert u == 1
assert v == 2
