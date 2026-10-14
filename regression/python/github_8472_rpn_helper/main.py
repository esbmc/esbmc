# A list of floats and strings reached through another list parameter is read
# through each element's runtime tag, as a direct argument already is (#8472).
def count_floats(xs):
    n = 0
    for x in xs:
        if isinstance(x, float):
            n += 1
    return n


def t(d):
    return count_floats(d[0])


assert t([[1.0, "+"]]) == 1
assert count_floats([1.0, "+"]) == 1


def t2(d):
    return count_floats(d[0]) + d[1]


assert t2([[1.0, "+", 2.0], 3]) == 5


def at(d, i):
    return count_floats(d[i])


assert at([[1.0, "+"], ["-", 2.0, 3.0]], 1) == 2
