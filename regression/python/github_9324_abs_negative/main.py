# abs(i) % 2 and a negative literal index.
def f():
    return 1


def g():
    return "a"


fs = [f, g]
i = nondet_int()
r = fs[abs(i) % 2]()
last = fs[-1]()
assert last == "a"
assert (r == 1) == (abs(i) % 2 == 0)
