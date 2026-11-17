def first_positive(values):
    for i in range(len(values)):
        if values[i] > 0:
            return list(values)
    return None


def size(p):
    return len(p)


r = first_positive([0])
assert size(r) == 1
