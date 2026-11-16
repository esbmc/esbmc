# GitHub #9936: two ints from imported functions are not refused
import mod

if nondet_bool():
    c = mod.fi()
else:
    c = mod.gi()
assert isinstance(c, int)
assert c == 42 or c == 7
