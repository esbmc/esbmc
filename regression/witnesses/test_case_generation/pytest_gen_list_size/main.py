def check(xs: list[int]) -> int:
    if len(xs) > 0:
        return 1
    return 2


check(nondet_list(2))
