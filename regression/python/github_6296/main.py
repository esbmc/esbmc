# An aliased module attribute must resolve like the unaliased one (#6296).
import math as m
import math

assert m.pi > 3.14
assert m.pi < 3.15
assert math.pi > 3.14


def normalise(angle: float) -> float:
    while angle >= 2 * m.pi:
        angle -= 2 * m.pi
    return angle


assert normalise(1.0) == 1.0
