from dataclasses import dataclass


@dataclass
class Base:
    x: int

    def __post_init__(self):
        self.x = self.x + 1


@dataclass
class Child(Base):
    y: int

    def __post_init__(self):
        super().__post_init__()
        self.y = self.y + self.x


c = Child(2, 3)
assert c.x == 3
assert c.y == 6
