r = 0
if "":
    r = 2
assert r == 0

if "a":
    r = 3
assert r == 3

if not "":
    r = 4
assert r == 4

x = 1 if "" else 5
assert x == 5
b = not "ab"
assert b == False
