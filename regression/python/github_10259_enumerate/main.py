def total(xs: list[int]) -> int:
    t: int = 0
    for i, v in enumerate(xs):
        t = t + i * v
    for p in enumerate(xs, 1):
        t = t + p[0] * p[1]
    return t


assert total([1, 2, 3]) == 22
