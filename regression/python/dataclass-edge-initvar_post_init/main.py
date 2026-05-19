from dataclasses import InitVar, dataclass


@dataclass
class Circle:
    radius: float
    diameter: float = 0.0
    scale: InitVar[float] = 1.0

    def __post_init__(self, scale: float):
        self.diameter = self.radius * 2.0 * scale


c = Circle(5.0, scale=2.0)
assert c.radius == 5.0
assert c.diameter == 20.0
