def f(xs: list[int]) -> int:
    i: int = 0
    return min(len(xs), i + 1)

assert f([1]) >= 0
