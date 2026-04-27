from dataclasses import dataclass, field

def make() -> int:
    return 100


@dataclass
class C:
    x: int = field(default_factory=make)


c = C(x=200)
# Factory wins; kwarg silently dropped.
assert c.x == 100
