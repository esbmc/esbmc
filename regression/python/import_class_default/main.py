# A constructor in an imported module gets its defaults filled too, not just a
# plain function's: each module is preprocessed on its own, so `Point()` used
# to be converted with no arguments at all and the attributes were never set.
from helper import Point, Acc

p = Point()
assert p.x == 5
assert p.y == 7

q = Point(1)
assert q.x == 1
assert q.y == 7

r = Point(1, 2)
assert r.x == 1
assert r.y == 2

# A container default reaches the constructor as a real list.
assert Acc().total == 0
assert Acc([1, 2, 3]).total == 3
