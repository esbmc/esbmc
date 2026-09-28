# A classmethod reached through self.make() constructs the subclass it runs on (#7947).
class Base:
    def __init__(self, x):
        self.x = x

    @classmethod
    def make(cls, x):
        return cls(x)

    def build(self):
        return self.make(3)


class Sub(Base):
    def __init__(self, x):
        self.x = x * 2


s = Sub(1)
t = s.build()
assert t.x == 3
