# Dataclass field access under --python-irep2-adjust: fields are struct
# components the adjuster resolves like any other member source.
from dataclasses import dataclass


@dataclass
class Point:
    x: int
    y: int


def main() -> None:
    p = Point(3, 4)
    assert p.x == 3
    assert p.y == 4


main()
