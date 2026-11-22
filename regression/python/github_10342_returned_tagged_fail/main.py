# r holds an int on the path where c is false.
from helper import pick

c = nondet_bool()
h = pick(c)
r = h()
assert isinstance(r, str)
