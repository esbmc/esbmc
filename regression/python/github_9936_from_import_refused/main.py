# GitHub #9936: an int or a float from imported functions, by path
from mod import fi, ff

if nondet_bool():
    c = fi()
    assert isinstance(c, int)
else:
    c = ff()
