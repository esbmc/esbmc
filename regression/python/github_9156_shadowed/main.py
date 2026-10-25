# Issue #9156: a parameter or local that shadows a module-level name holding a
# str or an int keeps its own type when returned.
c = nondet_bool()
if c:
    r = "a"
else:
    r = 1


def k():
    r = [1, 2]
    return r


def h(r: list):
    return r


assert k() == [1, 2]
assert h([3]) == [3]
