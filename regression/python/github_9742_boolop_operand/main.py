# GitHub #9742: and/or return one of their operands, not a bool.
def f(x, y):
    return x and y


def g(x, y):
    return x or y


assert f([], [1]) == []
assert f([3], [1, 2]) == [1, 2]
assert g([], [1, 2]) == [1, 2]
assert g([5], [1]) == [5]
assert f(2, 3) == 3
assert g("", "ab") == "ab"
