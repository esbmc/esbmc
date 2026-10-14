# shapes binds Shape as a parameter too, so a rename would change
# make(Shape=5); the collision stays refused (#7397).
import shapes
class Shape:
    def __init__(self) -> None:
        self.edges: int = 3
    def total(self) -> int:
        return self.edges

assert shapes.make(Shape=5) == 5
