import m
from m import func, C

r = m.func()
assert r is None
assert func() is None
c = C()
assert c.bump() is None and c.x == 2
