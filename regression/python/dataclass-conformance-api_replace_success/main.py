from dataclasses import dataclass, replace


@dataclass
class Point:
    x: int
    y: int


p = Point(1, 2)
q = replace(p, y=9)
assert p.x == 1 and p.y == 2
assert q.x == 1 and q.y == 9
