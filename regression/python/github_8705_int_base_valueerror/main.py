n = int("ff", 16)
assert n == 255
ok = False
try:
    int("zz", 16)
except ValueError:
    ok = True
assert ok
