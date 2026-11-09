def mk(e):
    return [5, 6] if e else None


c = mk(True)
assert len(c) == 2
assert c[1] == 6
