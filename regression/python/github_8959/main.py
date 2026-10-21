def my_sum(a, *rest):
    r = a
    for x in rest:
        r += x
    return r


f = my_sum


def use(n: int) -> int:
    return f(n, 1) + f(n)


assert f(1, 2, 3) == 6
assert f(1) == 1
assert f(1, *[2, 3]) == 6
assert use(4) == 9
