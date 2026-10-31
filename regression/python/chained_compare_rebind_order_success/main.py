g: int = 1


def bump() -> int:
    global g
    g = 5
    return 1


assert 0 < g + bump() < 3
assert g == 5

g = 1
assert 0 < 1 < g + bump() < 3
assert g == 5

g = 1
assert not (3 < 1 < g + bump() < 7)
assert g == 1
