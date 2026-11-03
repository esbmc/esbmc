def f(arr):
    first, second = arr
    return first


def g(arr):
    head, *rest = arr
    return head


def t(d):
    return f(d[0])


def u(d):
    r = g(d[0])
    return r


def h(arr):
    first, second = arr
    return first


def v(d):
    return h(d[0])


# Each wrong assertion sits on its own path, so each is reported
k = nondet_int()
if k == 0:
    assert t([["a", "b"]]) == "b"
elif k == 1:
    assert u([["c", "d", "e"]]) == "d"
else:
    assert v([[1, 2]]) == 2
