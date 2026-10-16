def append_one(a):
    return a + [1]


def concat(a, b):
    return a + b


def twice(a):
    return a * 2


def zeros(n):
    return n * [0]


def append_tuple(t):
    return t + (1,)


assert append_one([2]) == [2, 1]
assert concat([1], [2]) == [1, 2]
assert twice([1]) == [1, 1]
assert zeros(2) == [0, 0]
assert append_tuple((2,)) == (2, 1)
