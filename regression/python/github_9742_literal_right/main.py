def f(x):
    return x and [1, 2]


assert f([]) == []
assert f([3]) == [1, 2]
