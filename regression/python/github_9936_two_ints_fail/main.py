# GitHub #9936: two ints from imported functions are not refused
import mod

if nondet_bool():
    c = mod.fi()
else:
    c = mod.gi()
assert c > 42
