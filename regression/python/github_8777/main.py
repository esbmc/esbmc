def second(t):
    return t[1]


def size(t):
    return len(t)


def kind(t):
    return isinstance(t, tuple) and not isinstance(t, list)


def total(t):
    a, b, c = t
    return a + b + c


x = tuple(range(3))
assert isinstance(x, tuple)
assert not isinstance(x, list)
assert second(tuple([5, 6])) == 6
assert size(tuple(range(3))) == 3
assert kind(tuple([1, 2]))
assert total(tuple(range(3))) == 3
y = [1, 2]
assert isinstance(y, list) and not isinstance(y, tuple)
assert isinstance(list(x), list) and not isinstance(list(x), tuple)
