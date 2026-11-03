def count_floats(xs):
    n = 0
    for x in xs:
        if isinstance(x, float):
            n += 1
    return n


def t(d):
    return count_floats(d[0])


assert t([[1.0, "+"]]) == 2
