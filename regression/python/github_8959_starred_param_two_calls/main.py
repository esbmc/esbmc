def my_sum(a, *rest):
    r = a
    for x in rest:
        r += x
    return r


def add3(a, b, c):
    return a + b + c


def call(f, a, extra):
    return f(a, *extra)


def call_add3(a, more):
    return add3(a, *more)


assert call(my_sum, 1, (2, 3)) == 6
assert call(my_sum, 10, (20, 30)) == 60
assert call_add3(1, [2, 3]) == 6
