# Appending to the list leaves a function the group does not know.
def f():
    return 1


def g():
    return "a"


def h():
    return 2.5


fs = [f, g]
fs.append(h)
i = nondet_int()
r = fs[i % 3]()
