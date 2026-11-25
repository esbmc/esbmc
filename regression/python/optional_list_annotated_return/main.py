def first_positive(values) -> list:
    for i in range(len(values)):
        if values[i] > 0:
            return list(values)
    return None


r = first_positive([1, 2])
assert r is not None and r[0] == 1 and len(r) == 2
assert first_positive([0]) is None
