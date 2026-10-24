g = 1


def bump() -> int:
    global g
    g = 10
    return 5


flag = False
ok = flag or g + bump() == 15
assert ok
