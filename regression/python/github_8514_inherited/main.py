class Base:

    def __init__(self):
        self.n = 2

    def count(self) -> int:
        return self.n


class Sub(Base):

    def upper(self) -> int:
        return self.count() + 1


s = Sub()
assert s.count() == 2
assert s.upper() == 3
t = "banana"
assert t.count("a") == 3
assert t.upper() == "BANANA"
