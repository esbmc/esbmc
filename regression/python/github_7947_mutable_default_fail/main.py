# An inherited classmethod with a mutable default is rejected rather than verified (#7947).
class Base:
    def __init__(self, n):
        self.n = n

    @classmethod
    def make(cls, acc=[]):
        acc.append(1)
        return cls(len(acc))


class Sub(Base):
    pass


b = Base.make()
s = Sub.make()
assert s.n == 1
