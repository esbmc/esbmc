# A list parameter holding a list and an int: a constant index reads each
# element at its own type, so the inner list can be subscripted (#8472).
def t(d):
    a = d[0]
    return a[1]


def second(d):
    return d[1] + len(d)


assert t([[1, 2], 1]) == 2
assert second([[1, 2], 1]) == 3
