def my_sum(a, *rest):
    r = a
    for x in rest:
        r += x
    return r


def call(f, a, extra):
    return f(a, *extra)


assert call(my_sum, 1, [2, 3]) == 6
assert call(my_sum, 1, [2]) == 3
