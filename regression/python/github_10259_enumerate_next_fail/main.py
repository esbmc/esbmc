def total(xs: list[int]) -> int:
    t: int = 0
    for i, v in enumerate(xs):
        t = t + xs[i + 1]
    return t


total([1, 2, 3])
