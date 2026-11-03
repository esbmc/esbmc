def bump(flag: bool) -> int:
    v = None
    if flag:
        v = 3
    return v + 1


bump(True)
bump(False)
