# += grows the list.
def f():
    return 1


def g():
    return "a"


fs = [f, g]
fs += [f]
i = nondet_int()
r = fs[i % 3]()
