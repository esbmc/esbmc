class Base:

    def __init__(self):
        self.n = 2

    def count(self) -> int:
        return self.n


class Sub(Base):
    pass


s = Sub()
t = "banana"
assert t.count("a") == 3
assert s.count() == 3
