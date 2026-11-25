def first_positive(values):
    for i in range(len(values)):
        if values[i] > 0:
            return list(values)
    return None


def any_positive(values):
    for x in values:
        if x > 0:
            return values


def size(p):
    return len(p)


r = first_positive([1, 2])
assert r is not None
assert r[0] == 1
alias = r
assert alias[1] == 2
assert len(r) == 2
assert size(r) == 2
total = 0
for x in r:
    total += x
assert total == 3
assert r == [1, 2]
assert any_positive([3]) == [3]
assert any_positive([0]) is None
