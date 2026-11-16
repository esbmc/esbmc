# key may be 0, so ls[0]() calls pick with no argument: TypeError.
def f():
    return 1


def g():
    return "a"


def pick(key):
    return ls[key]()


ls = [pick, f, g]
key = nondet_int() % 3
r = pick(key)
