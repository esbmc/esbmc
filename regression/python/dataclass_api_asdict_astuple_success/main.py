from dataclasses import dataclass, asdict, astuple


@dataclass
class Child:
    v: int


@dataclass
class Parent:
    c: Child


p = Parent(Child(7))

d = asdict(p)
t = astuple(p)

assert d["c"]["v"] == p.c.v
assert t[0][0] == p.c.v
