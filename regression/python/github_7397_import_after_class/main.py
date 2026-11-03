# The import rebinds Shape before anything reads the local class, so every
# use of Shape is the imported class, as in CPython (#7397, #8343).
class Shape:
    def __init__(self) -> None:
        self.edges: int = 3
    def total(self) -> int:
        return self.edges

from shapes import Shape

s = Shape()
assert s.total() == 4
