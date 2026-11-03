# The guarded call returns a list, while the arity error is typed int.
def pair(a, b):
    return [b, a]


def check(xs, expected):
    assert pair(*xs) == expected


check([1, 2], [2, 1])
check([3, 4], [4, 3])
