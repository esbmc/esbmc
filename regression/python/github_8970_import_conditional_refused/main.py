# A name imported under a condition is not dispatched.
from helper import g

if nondet_bool():
    from helper import f
    h = f
else:
    h = g
r = h()
