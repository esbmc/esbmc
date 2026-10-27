# GitHub #9088: list + tuple raises TypeError unless the list was built by
# tuple(x), which is itself a tuple.

try:
    x = [1] + (2, 3)
    assert False
except TypeError:
    pass

try:
    y = (2, 3) + [1]
    assert False
except TypeError:
    pass

try:
    z = (2, 3) + 1
    assert False
except TypeError:
    pass

t = tuple([1])
u = t + (2, 3)
assert len(u) == 3
assert u[0] == 1 and u[1] == 2 and u[2] == 3
assert isinstance(u, tuple)
v = (5, 6) + t
assert len(v) == 3
assert v[0] == 5 and v[2] == 1

n = 0


def a() -> tuple[int]:
    global n
    n = n * 10 + 1
    return (1,)


def b() -> list[int]:
    global n
    n = n * 10 + 2
    return tuple([2])


r = a() + b()
assert n == 12
assert len(r) == 2 and r[0] == 1 and r[1] == 2


def lazy(l: list[int], c: bool) -> int:
    t = tuple(l)
    x = c and len(t + (2, )) == 2
    return 1 if x else 0


assert lazy([1], True) == 1
assert lazy([1], False) == 0
