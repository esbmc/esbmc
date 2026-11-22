def total(xs: list[int]) -> int:
    s: int = 0
    for x in xs:
        s = s + x
    return s + xs[3]


assert total([1, 2, 3]) == 6
