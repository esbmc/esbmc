def first_positive(values):
    for i in range(len(values)):
        if values[i] > 0:
            return list(values)
    return None


r = first_positive([0])
for x in r:
    assert x == 0
