def head(t: tuple[int, int], d: int) -> int:
    first, second = t
    if d:
        head((5, 6), d - 1)
    return first


assert head((1, 2), 1) == 5
