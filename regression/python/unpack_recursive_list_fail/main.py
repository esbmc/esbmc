def star_head(arr: list[int], d: int) -> int:
    first, *rest = arr
    if d:
        star_head(rest, d - 1)
    return first


assert star_head([1, 2, 3], 2) == 3
