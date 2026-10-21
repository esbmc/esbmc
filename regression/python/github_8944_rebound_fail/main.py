def pick(c: bool) -> int:
    n = 2
    if c:
        n = 3
    return 5**n


assert pick(False) == 125
