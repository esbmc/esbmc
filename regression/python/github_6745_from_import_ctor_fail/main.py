from shapes import Shape as Base


class Square(Base):

    def read_sides(self) -> int:
        return self.sides


s = Square()
assert s.read_sides() == 5
