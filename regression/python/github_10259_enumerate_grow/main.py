def count(xs: list[int]) -> int:
    n: int = 0
    for i, v in enumerate(xs):
        if i < 2:
            xs.append(v)
        n = n + 1
    return n


assert count([1, 2]) == 4
