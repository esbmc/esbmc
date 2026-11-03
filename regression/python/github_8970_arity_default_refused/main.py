# other has a default, so ls[0]() is a valid call and an arm that raises would be wrong.
def f():
    return 1


def g():
    return "a"


def other(x=0):
    return 2.5


def pick(key):
    return ls[key]()


ls = [other, f, g]
key = nondet_int() % 3
r = pick(key)
