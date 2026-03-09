def identity(x: int) -> int:
    return x

def f(xs: list[int]) -> int:
    n: int = 5
    return min(len(xs), identity(n))

assert f([1, 2]) == 2
assert f([1, 2, 3, 4, 5, 6]) == 5
