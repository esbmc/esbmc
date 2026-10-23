def my_sum(a, *rest):
    r = a
    for x in rest:
        r += x
    return r


f = my_sum
