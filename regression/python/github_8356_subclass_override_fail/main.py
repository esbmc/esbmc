class Base:
    def scale(self, k: int) -> int:
        return k * 2

    times = scale


class Sub(Base):
    def scale(self, k: int) -> int:
        return k * 3


s = Sub()
assert s.times(5) == 15
