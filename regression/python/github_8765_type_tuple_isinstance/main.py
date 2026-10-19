ok = False
for a in (str, int):
    if isinstance(1, a):
        ok = True
assert ok
