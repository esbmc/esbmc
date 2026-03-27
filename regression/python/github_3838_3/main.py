# Test: super().method() result used in arithmetic
class Shape:
    def area(self) -> int:
        return 10

class Square(Shape):
    def area(self) -> int:
        base_area: int = super().area()
        return base_area * 2

s = Square()
assert s.area() == 20
