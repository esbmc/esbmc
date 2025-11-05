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
class A:
    def __init__(self, val: int) -> None:
        self.val: int = val

    def create_b(self) -> 'B':
        return B(self)

class B:
    def __init__(self, a: A) -> None:
        self.val: int = a.val

    def create_c(self) -> 'C':
        return C(self)

class C:
    def __init__(self, b: B) -> None:
        self.val: int = b.val

a = A(7)
b = a.create_b()
c = b.create_c()
assert c.val == 7