# Multiple instances: instance attribute set via function, class attr unchanged
class Point:
    x = 0
    y = 0

def move(p):
    p.x = 10
    p.y = 20

p1 = Point()
p2 = Point()
move(p1)
assert p1.x == 10
assert p1.y == 20
assert p2.x == 0
assert p2.y == 0
