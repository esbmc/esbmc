g: int = 1


def bump() -> int:
    global g
    g = 5
    return 1


h = lambda: g + bump()
assert h() == 6
