from dataclasses import dataclass, replace


@dataclass
class C:
    x: int
    y: int


c1 = C(1, 2)
c2 = replace(c1, y=9)

assert c1.x == 1
assert c1.y == 2
assert c2.x == 1
assert c2.y == 9
