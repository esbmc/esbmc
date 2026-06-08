from dataclasses import dataclass


@dataclass
class Base:

    def __post_init__(self) -> None:
        self.order = 1


@dataclass
class Child(Base):
    value: int


child = Child(9)

assert child.order == 1
assert child.value == 9
