def f(x: tuple[tuple[list[int], int], int]) -> int:
    ([a, b], c), d = x
    return a * b + c * d


assert f((([2, 3], 4), 5)) == 26
