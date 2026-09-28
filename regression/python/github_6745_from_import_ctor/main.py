from shapes import Shape


class Square(Shape):

    def read_sides(self) -> int:
        return self.sides


s = Square()
assert s.read_sides() == 4
