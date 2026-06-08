from dataclasses import dataclass


@dataclass
class Base:

    def __post_init__(self) -> None:
        self.order = 1


@dataclass
class Child(Base):

    def __post_init__(self) -> None:
        Base.__post_init__(self)
        self.order = self.order + 1


child = Child()

assert child.order == 2
