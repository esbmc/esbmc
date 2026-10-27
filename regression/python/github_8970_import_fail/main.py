# As github_8970_import, but the int path asserts what is false.
from helper import f, g

c = nondet_bool()
h = f if c else g
r = h()
if c:
    assert r == "a"
else:
    assert r + 1 == 3
