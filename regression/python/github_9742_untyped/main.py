def f(x, y):
    return x and y


assert f([], [1]) == []
assert f([2], [3, 1]) == [3, 1]
