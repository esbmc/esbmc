# A function chosen by a conditional expression, passed as an argument
# (#8475).
def f():
    return 1


def g():
    return 2


def call(a):
    return a()


c = call(f if nondet_bool() else g)
assert c == 1
