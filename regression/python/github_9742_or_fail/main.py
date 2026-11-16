def f(x: list[int], y: list[int]):
    return x or y


assert f([2], [1]) == [1]
