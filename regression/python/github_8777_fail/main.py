def second(t):
    return t[1]


def kind(t):
    return isinstance(t, list)


def total(t):
    a, b, c = t
    return a + b + c


# Each wrong assertion sits on its own path, so each is reported
k = nondet_int()
x = tuple(range(3))
if k == 0:
    assert isinstance(x, list)
elif k == 1:
    assert second(tuple([5, 6])) == 5
elif k == 2:
    assert kind(tuple([1, 2]))
elif k == 3:
    assert total(tuple(range(3))) == 4
elif k == 4:
    assert isinstance(list(x), tuple)
else:
    assert isinstance([1, 2], tuple)
