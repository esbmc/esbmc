def total(base, *rest):
    s = base
    for v in rest:
        s = s + v
    return s


assert total(1, 2, 3) == 7
