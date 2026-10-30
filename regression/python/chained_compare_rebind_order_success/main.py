g: int = 1


def bump() -> int:
    global g
    g = 5
    return 1


assert 0 < g + bump() < 3
assert g == 5
