# An imported name that its module defines twice is not dispatched.
from helper import f, g

h = f if nondet_bool() else g
r = h()
