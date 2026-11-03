# The callee is a parameter, not the global add, so the global's parameter
# count does not apply.
def add(a, b):
    return a + b


def add3(a, b, c):
    return a + b + c


def run(add, xs):
    return add(*xs)


assert run(add3, [1, 2, 3]) == 6
