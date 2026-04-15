# Test __str__ with f-string using multiple member fields
class Point:
    def __init__(self, x: str, y: str):
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return f"Point({self.x}, {self.y})"

p = Point("10", "20")
assert str(p) == "Point(10, 20)"
