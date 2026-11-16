def mk(e):
    return None if e else [5]

c = mk(True)
assert c[0] == 5
