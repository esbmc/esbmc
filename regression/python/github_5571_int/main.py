def f(pairs: list[tuple[int, int]]) -> int:
    s = 0
    for u, v in pairs:
        s = s + v
    return s


assert f([(1, 2), (3, 4)]) == 6
