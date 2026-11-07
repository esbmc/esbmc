# GitHub #9936: an int or a float from imported functions, by path
import mod

if nondet_bool():
    c = mod.fi()
    assert isinstance(c, int)
else:
    c = mod.ff()
