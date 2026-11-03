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


assert t([["a", "b"]]) == "a"
assert u([["c", "d", "e"]]) == "c"
assert v([[1, 2]]) == 1
