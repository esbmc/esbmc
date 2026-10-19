def one():
    return 1


def two():
    return 2


def call(a):
    return a()


x = call(one) if nondet_bool() else call(two)
y = x + 1
assert y == 2 or y == 3
