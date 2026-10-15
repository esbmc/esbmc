def total(pairs: list[tuple[int, int]]) -> int:
    t = 0
    for i, (a, b) in enumerate(pairs, start=1):
        t += i * (a + b)
    return t


assert total([(1, 2), (3, 4)]) == 16
