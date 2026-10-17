# A call through a function an imported function returns (#8351).
from mod import pick

g = pick(nondet_bool())
r = g()
assert r == 1
