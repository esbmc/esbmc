def f(x: list[int], y: list[int], z: list[int]):
    return x and y and z


assert f([1], [2], [3]) == [3]
assert f([1], [], [3]) == []
assert f([], [2], [3]) == []
