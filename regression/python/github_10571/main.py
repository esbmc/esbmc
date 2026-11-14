import harness


def func(x):
    return isinstance(x, str)


s = harness.nondet_str() or None
x, y = map(func, [1, s])
assert not x
