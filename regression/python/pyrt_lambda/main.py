# A lambda is an ordinary function under a name of its own, so it can be
# bound, called, or handed to something that calls it.
f = lambda x: x + 1
assert f(1) == 2

g = lambda a, b: a * b
assert g(3, 4) == 12

h = lambda: 7
assert h() == 7


def apply(fn, v):
    return fn(v)


assert apply(lambda x: x * 2, 5) == 10
