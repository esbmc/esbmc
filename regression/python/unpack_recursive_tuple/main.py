def head(t: tuple[int, int], d: int) -> int:
    first, second = t
    if d:
        head((5, 6), d - 1)
    return first


def star_head(t: tuple[int, int, int], d: int) -> int:
    first, *rest = t
    if d:
        star_head((5, 6, 7), d - 1)
    return first + len(rest)


assert head((1, 2), 1) == 1
assert star_head((1, 2, 3), 1) == 3
