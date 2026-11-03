# The call passes a variable, whose evaluation the raising arm would skip.
def f():
    return 1


def g():
    return "a"


def pick(key, x):
    return ls[key](x)


ls = [pick, f, g]
key = nondet_int() % 3
r = pick(key, 1)
