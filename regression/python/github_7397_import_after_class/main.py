# The import rebinds Shape after the local class, so the rename would bind
# the wrong class; the collision stays refused (#7397).
class Shape:
    def __init__(self) -> None:
        self.edges: int = 3
    def total(self) -> int:
        return self.edges

from shapes import Shape

s = Shape()
assert s.total() == 4
