# The list is rebound to ints on one path.
def f():
    return 1


def g():
    return "a"


fs = [f, g]
if nondet_bool():
    fs = [1, 2]
i = nondet_int()
r = fs[i % 2]()
