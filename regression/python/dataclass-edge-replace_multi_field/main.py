from dataclasses import dataclass, replace


@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int


r = Rect(0, 0, 10, 20)
r2 = replace(r, x=5, y=3)
assert r2.x == 5
assert r2.y == 3
assert r2.w == 10
assert r2.h == 20
assert r.x == 0
