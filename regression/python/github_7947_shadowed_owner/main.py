# A classmethod with a local named after its class is rejected rather than verified (#7947).
class Foo:
    def __init__(self, x):
        self.x = x

    @classmethod
    def make(cls, v):
        Foo = v + 1
        return cls(Foo)


m = Foo.make(5)
assert m.x == 6
