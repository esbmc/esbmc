class Point:
    def __init__(self, x: int, y: int) -> None:
        self.x: int = x
        self.y: int = y
    
    def copy_to_vector(self) -> 'Vector':
        return Vector(self)

class Vector:
    def __init__(self, p: Point) -> None:
        self.x: int = p.x
        self.y: int = p.y

p = Point(3, 4)
v = p.copy_to_vector()
assert v.x == 3
assert v.y == 4

