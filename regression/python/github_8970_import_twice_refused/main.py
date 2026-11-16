# A name that the program imports twice is not dispatched.
from helper import f, g
from other import f

h = f if nondet_bool() else g
r = h()
