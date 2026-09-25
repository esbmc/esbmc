from first import X as Base
from second import Base


class Square(Base):

    def read_sides(self) -> int:
        return self.sides


s = Square()
assert s.read_sides() == 4
