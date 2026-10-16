def bump(flag: bool) -> int:
    v = None
    if flag:
        v = 3
    if v is None:
        return 0
    return v + 1


assert bump(True) == 4
assert bump(False) == 0
