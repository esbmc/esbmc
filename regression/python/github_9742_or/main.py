def f(x: list[int], y: list[int]):
    return x or y


assert f([], [1]) == [1]
assert f([2], [1]) == [2]
