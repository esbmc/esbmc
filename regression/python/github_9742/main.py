def f(x: list[int], y: list[int]):
    return x and y


assert f([], [1]) == []
assert f([2], [1]) == [1]
