# key reaches 3, which is out of range for ls: IndexError.
def f():
    return 1


def g():
    return "a"


def pick(key):
    return ls[key]()


ls = [pick, f, g]
key = nondet_int() % 3 + 1
r = pick(key)
