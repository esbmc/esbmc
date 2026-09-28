# An inherited classmethod with a local named after the subclass is rejected rather than verified (#7947).
class Base:
    def __init__(self, x):
        self.x = x

    @classmethod
    def make(cls, v):
        Sub = v + 1
        return cls(Sub)


class Sub(Base):
    pass


s = Sub.make(5)
assert s.x == 6
