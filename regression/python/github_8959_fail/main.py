def my_sum(a, *rest):
    r = a
    for x in rest:
        r += x
    return r


f = my_sum
assert f(1, 2, 3) == 7
